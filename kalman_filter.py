

import numpy as np



# TODO Part 3: Comment the code explaining each part
class kalman_filter:
    
    # TODO Part 3: Initialize the covariances and the states    
    def __init__(self, P,Q,R, x, dt):
        
        self.P=P # State covariance matrix
        self.Q=Q # Process noise covariance matrix
        self.R=R # Measurement noise covariance matrix
        self.x=x # Initial state estimate
        self.dt =dt # Time step
        
    # TODO Part 3: Replace the matrices with Jacobians where needed        
    def predict(self):

        self.A = self.jacobian_A() # Motion model
        self.C = self.jacobian_H() # Measurement model
        
        self.motion_model()
        
        self.P= np.dot( np.dot(self.A, self.P), self.A.T) + self.Q

    # TODO Part 3: Replace the matrices with Jacobians where needed
    def update(self, z):

        # Get Kalman gain
        S=np.dot(np.dot(self.C, self.P), self.C.T) + self.R
            
        kalman_gain=np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))
        
        # Calculate measurement error
        surprise_error= z - self.measurement_model()
        
        # Update prediction using error
        self.x=self.x + np.dot(kalman_gain, surprise_error)
        self.P=np.dot( (np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.C)) , self.P)
        
    
    # TODO Part 3: Implement here the measurement model
    def measurement_model(self):
        x, y, th, w, v, vdot = self.x
        return np.array([
            v,# v
            w,# w
            vdot, # ax
            v*w, # ay
        ])
        
    # TODO Part 3: Impelment the motion model (state-transition matrice)
    def motion_model(self):
        
        x, y, th, w, v, vdot = self.x
        dt = self.dt
        
        self.x = np.array([
            x + v * np.cos(th) * dt, # Update x position
            y + v * np.sin(th) * dt, # Update y position
            th + w * dt, # Update angle
            w, # Assume constant angular velocity
            v  + vdot*dt, # Update linear velocity
            vdot # Assume constant linear acceleration
        ])
        

    def jacobian_A(self):
        x, y, th, w, v, vdot = self.x
        dt = self.dt

        return np.array([
            #x, y,               th, w,             v, vdot
            [1, 0,  -v*np.sin(th)*dt, 0, np.cos(th)*dt,  0], # Derivative of x wrt theta and v
            [0, 1,   v*np.cos(th)*dt, 0, np.sin(th)*dt,  0], # Derivative of y wrt theta and v
            [0, 0,                1, dt,           0,  0],
            [0, 0,                0, 1,            0,  0],
            [0, 0,                0, 0,            1,  dt],
            [0, 0,                0, 0,            0,  1 ]
        ])
    
    
    # TODO Part 3: Implement here the jacobian of the H matrix (measurements)    
    def jacobian_H(self):
        x, y, th, w, v, vdot=self.x
        return np.array([
            #x, y,th, w, v,vdot
            [0,0,0  , 0, 1, 0], # v
            [0,0,0  , 1, 0, 0], # w
            [0,0,0  , 0, 0, 1], # ax
            [0,0,0  , v, w, 0], # ay
        ])
        
    # TODO Part 3: return the states here    
    def get_states(self):
        return self.x
