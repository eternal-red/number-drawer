import numpy as np

y_pred = np.asarray([0.4, 0.3, 0.15, 0.15])
n, e, s, w = np.asarray([0, 1]), np.asarray([1, 0]), np.asarray([0, -1]), np.asarray([-1, 0])
some_dir = [n, e, s, w]

vecs = [out*dir for out, dir, in zip(y_pred, some_dir)]
vec_sum = sum(vecs)
all_dir = [(n+w)/np.sqrt(2), n, (n+e)/np.sqrt(2), e, (e+s)/np.sqrt(2), s, (s+w)/np.sqrt(2), w]
print(np.argmax([np.dot(vec_sum, dir) for dir in all_dir]) + 1)
