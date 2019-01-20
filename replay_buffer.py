import numpy as np

class ReplayBuffer:
	'''
	A replay buffer implementation to store the
	most size recent experiences D=(s,a,r,s',d) of the agent
	'''
	def __init__(self, size, state_dim, action_dim):
		self._s_buff  = np.zeros([size, state_dim],  dtype=np.float32)
		self._a_buff  = np.zeros([size, action_dim], dtype=np.float32)
		self._r_buff  = np.zeros(size,               dtype=np.float32)
		self._s2_buff = np.zeros([size, state_dim],  dtype=np.float32)
		self._d_buff  = np.zeros(size,               dtype=np.float32)
		self.ptr, self.size, self.max_size = 0, 0, size

	def store(self, s, a, r, s2, d):
		self._s_buff[self.ptr]  = s
		self._a_buff[self.ptr]  = a
		self._r_buff[self.ptr]  = r
		self._s2_buff[self.ptr] = s2
		self._d_buff[self.ptr]  = d
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size+1, self.max_size)

	def sample_batch(self, batch_size=32):
		'''
		Returns a dictionary of batch_size size of random 
		experiencies
		'''
		idx = np.random.randint(0, self.size, batch_size)
		return dict(s=self._s_buff[idx],
                            a=self._a_buff[idx],
                            r=self._r_buff[idx],
                            s2=self._s2_buff[idx],
                            d=self._d_buff[idx])
