import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

class Source(gym.Env):
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        # r_e is the radius that encloses half of the total light of the galaxy, set here between 0 and 1 (scaled to image size)
        # n [0.1, 15], r_e [0, 1], q [0,1], theta [0, 2pi], center_x [0,1], center_y [0,1]
        self.low = np.reshape(np.array([0., 0., 0., 0., -1., -1.,]), (1, -1)) * np.ones((self.B, 1))
        self.high = np.reshape(np.array([15., 1., 1., 2*np.pi, 1., 1.]), (1, -1)) * np.ones((self.B, 1))
        self.action_space = gym.spaces.Box(
            low = self.low,
            high = self.high,
            dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=-2*np.ones((self.B, self.image_x, self.image_y)), 
            high=2*np.ones((self.B, self.image_x, self.image_y)), 
            dtype=np.float64
        )

    def reset(self, source_labels):
        batch_mins = np.reshape(np.min(source_labels, axis=(-2, -1)), (-1, 1, 1))
        batch_maxs = np.reshape(np.max(source_labels, axis=(-2, -1)), (-1, 1, 1))
        self.source_labels = (source_labels - batch_mins) / (batch_maxs - batch_mins)
        self.sersic_list = []
        self.image_arcsec_bounds = np.array([self.image_x * self.arcsec_per_pixel / 2, self.image_y * self.arcsec_per_pixel / 2])
        return self.source_labels, self._get_info()

    def step(self, action, delta=0.1):
        action = self._process_actions(action)
        self.sersic_list.append(action)
        constructed_source = self._construct_sersics()
        source_diff = self.source_labels - constructed_source
        reward = -np.mean((source_diff)**2, axis=(-2, -1))
        done = np.mean(np.abs(source_diff), axis=(-2, -1)) < delta
        if self.render_mode == 'source':
            return source_diff, reward, done, False, {'source':constructed_source}  
        return source_diff, reward, done, False, {}

    def render(self, mode='human'):
        return
    
    def _init_hyperparameters(self, hyperparameters):
        self.B = 1
        self.image_x, self.image_y, self.image_c = 1, 1, 1
        self.seed = 0
        self.cuda = False
        self.arcsec_per_pixel = 0.001
        self.render_mode = 'none'
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available else 'cpu')
        print(f'[ENV] Using {self.device}')

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(f"[ENV] Seed set to {self.seed}")

    def _get_info(self):
        return {}
    
    def _process_actions(self, action):
        # actions given in [0, 1]
        action = np.clip(action * (self.high - self.low) + self.low, self.low + 1e-1, self.high)
        return action
    
    def _construct_sersics(self):
        constructed_source = np.zeros_like(self.source_labels)
        for sersic_models in self.sersic_list:
            n, r_e_, q, theta, centers_x, centers_y = sersic_models[:,0], sersic_models[:,1], sersic_models[:,2], sersic_models[:,3], sersic_models[:,4]*self.image_arcsec_bounds[0], sersic_models[:,5]*self.image_arcsec_bounds[1] # (B,)
            b_n = 2 * n - 0.331 
            r_e = np.linalg.norm(self.image_arcsec_bounds) * r_e_
            r_e = np.reshape(r_e, (-1, 1, 1)) # fit for broadcasting
            q = np.reshape(q, (-1, 1, 1))
            n = np.reshape(n, (-1, 1, 1))
            b_n = np.reshape(b_n, (-1, 1, 1))
            centers_x, centers_y = np.reshape(centers_x, (-1, 1, 1)), np.reshape(centers_y, (-1, 1, 1))

            # arcsec_bound = resolution*x/2
            pos_x = np.linspace(-self.image_arcsec_bounds[0], self.image_arcsec_bounds[0], self.image_x)
            pos_y = np.linspace(-self.image_arcsec_bounds[1], self.image_arcsec_bounds[1], self.image_y)
            phi_y, phi_x = np.meshgrid(pos_x, pos_y)
            phi_y, phi_x = np.reshape(phi_y, (1, self.image_y, self.image_x)), np.reshape(phi_x, (1, self.image_x, self.image_x))

            phi_y, phi_x = phi_y - centers_y, phi_x - centers_x # shifted coordinates in pixel scale (B, y, x,)
            cos_theta, sin_theta = np.reshape(np.cos(theta), (-1, 1, 1)), np.reshape(np.sin(theta), (-1, 1, 1)) # (B, 1, 1)
            x_rot, y_rot = phi_x * cos_theta + phi_y * sin_theta, -phi_x * sin_theta + phi_y * cos_theta # rotated coordinates in (fractional) pixel scale (B, y, x)
            # print(q)
            R = np.sqrt((x_rot**2)/q + q*(y_rot**2)) * self.arcsec_per_pixel # (B, y, x,) in arcsec
            R[R==0] = 1e-6
            I = np.exp(-b_n * ((R / r_e) ** (1/n) - 1)) # (B, y, x,)
            constructed_source += I
        constructed_source = (constructed_source - np.min(constructed_source, axis=(-2, -1), keepdims=True)) / (np.max(constructed_source, axis=(-2, -1), keepdims=True) - np.min(constructed_source, axis=(-2, -1), keepdims=True) + 1e-8) # minmax normalized
        return constructed_source
    