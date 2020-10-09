import numpy as np 
import matplotlib.pyplot as plt 
from kf import KF 

plt.ion()
plt.figure()
kf = KF(initial_x = 0.0, initial_v = 1.0, accel_variance= 0.1)

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS = 20

real_x = 0.0
meas_variance = 0.1**2
real_v = 0.9

moys = []
covs = []

for step in range(NUM_STEPS):
    covs.append(kf.cov)
    moys.append(kf.mean)

    real_x = real_x + DT*real_v

    kf.predict(dt = DT)
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value = real_x + np.random.randn() * np.sqrt(meas_variance), 
                    meas_variance = meas_variance)






plt.subplot(2,1,1)
plt.title("Position")
plt.plot([mu[0] for mu in moys], 'r')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(moys, covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(moys, covs)], 'r--')



plt.subplot(2,1,2)
plt.title("Velocity")
plt.plot([mu[1] for mu in moys], 'r')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(moys, covs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(moys, covs)], 'r--')


plt.show()
plt.ginput(1)
