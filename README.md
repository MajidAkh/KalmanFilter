# KalmanFilter
Kalman Filter

1.Design

The goal of this project is to be able to estimate the location of an object x at a speed v it could be moving in one or two or three dimension.

measurements we have is z = x + epsilon(r) which each a noisy location.

epsilon(r) follow a Normal distribution such that N(0, sigma²).

X_k = [x_k, v_k], 
Z_k = [z_k]

X_k+1 = K_k + V_k*delta(t) + 1/2*a*delta(t)²
V_k+1 = V_k + a*delta(t)

X_k 

	
2.Prototyping on Python

3. Porting on C++
