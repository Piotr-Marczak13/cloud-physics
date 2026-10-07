import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from IPython.display import HTML
plt.rcParams["animation.html"] = "jshtml"


#global variables
#bare MC movement
eps = 1e-3
L = 1.
k = (eps*L)**(2/3)
C0 = 2.1
tU = 4*k/(3*C0*eps)
sigma_u = np.sqrt(2*k/3)

#Molecular diffusivity of water vapor
Dv = 2.16e-5
#thermal conductivity
kappa = 2.38e-2

#thermodynamics
pressure = 85000  #Pa
eps2=0.662 #from paper 

tau_turb = 1 #?
e0 = 611 #Pa at triple point
T00 = 273.16 #K at triple point
#enthalpy of vaporization
Lv0 = 2.5e6  #J/kg
#Gas constant of water vapor
Rv = 461.5
omega = 1/tau_turb




def partialPressure(p,q):
    return p/(1+eps2/q)
def equiPressure(T):
    return e0*np.exp( (Lv0 / Rv) *(1/T00 - 1/T))
def superSat(p, T, q):
    return partialPressure(p,q)/equiPressure(T)-1

#organize the concepts into objects
class Gas:
    def __init__(self, nParticles, box_size, dt, nDimensions = 2):
        self.nDim = nDimensions
        self.nParticles = nParticles
        self.box_size = box_size
        self.dt = dt
        self.initialize_particles() #pos and vel

    def initialize_particles(self):
        # Initialize particle positions randomly within the box
        positions = np.random.rand(self.nParticles, self.nDim) * self.box_size
        
        # Initialize particle velocities based on the temperature
        velocities = np.random.normal(0, sigma_u, (self.nParticles, self.nDim))
        self.velocity_95cent = 2*sigma_u
        
        self.positions, self.velocities = positions, velocities
    
    def update_positions(self):
        #Update the position based on previous velocity
        self.positions += self.velocities * self.dt
        #Apply periodic boundary conditions
        self. positions = (self.positions + self.box_size) % self.box_size
        #Update the velocity based on the temperature difference
        self.velocities += -self.velocities/tU*self.dt + np.sqrt(2*sigma_u**2/tU)*np.random.normal(0, np.sqrt(self.dt), (self.nParticles, self.nDim))

    def run_bare(self,nSteps):
        self.initialize_particles()
        posHist = np.zeros((nSteps, self.nParticles, self.nDim))
        for i in tqdm(range(nSteps)):
            posHist[i] = self.positions
            self.update_positions()
        return posHist
    
    def animate(self, posHist):
        fig, ax = plt.subplots()
        scat = ax.scatter(posHist[0,:,0], posHist[0,:,1], s=10)
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        def update(frame):
            scat.set_offsets(posHist[frame])
            return scat,
        ani = animation.FuncAnimation(fig, update, frames=posHist.shape[0], interval=self.dt*1000, blit=True)
        plt.close(fig)
        return ani
    
    def initialize_thermo(self, Ct, Cq, Tin, Tout, Rin, Rout, QinCoeff = 1.1, QoutCoeff = 0.6):
        self.Rin = Rin
        self.Rout = Rout
        self.p = pressure
        self.Ct = Ct
        self.Cq = Cq
        self.Tin = Tin
        self.Tout = Tout
        self.Ts = np.zeros(self.nParticles) #temperature of each particle
        self.qs = np.zeros(self.nParticles) #humidity of each particle
        self.Ts = np.where((self.positions[:,0] > self.box_size/3) & (self.positions[:,0] < 2*self.box_size/3), Tin, Tout) #initial temperature based on position
        self.Qin=QinCoeff*eps2*(equiPressure(Tin)/(pressure-equiPressure(Tin))) 
        self.Qout=QoutCoeff*eps2*(equiPressure(Tout)/(pressure-equiPressure(Tout))) 
        self.qs = np.where((self.positions[:,0] > self.box_size/3) & (self.positions[:,0] < 2*self.box_size/3), self.Qin, self.Qout) #initial humidity based on position
        self.Tfixed = self.Ts.copy()#fixed temperature for relaxation
        self.qfixed = self.qs.copy() #fixed humidity for relaxation
        self.T_average = None #average temperature for average relaxation
        self.q_average = None #average humidity for average relaxation

        #Dynamic bin for average relaxation mode
        #nOfBins = max(10, self.nParticles // 100)
        nOfBins = 50
        self.T_average = np.zeros(nOfBins)
        self.q_average = np.zeros(nOfBins)

        #initialize radius of droplets
        print("Initializing R")
        self.initialize_R()
        print("Done")

    def update_thermo(self, relaxation_mode = "zones", mixing_mode = "IECM"):
        #Update the temperature and humidity based on the current state
        #Initialize thermo check
        if not hasattr(self, 'Ct'):
            print("Thermodynamics not initialized. Call initialize_thermo(Ct, Cq, Tin, Tout)) first.")
            return
        #Calculate temperatures for every particle
        if relaxation_mode == "fixed":
            boolArr = (self.positions[:,0] > self.box_size/3) & (self.positions[:,0] < 2*self.box_size/3)
            self.Tfixed = np.where(boolArr, self.Tin, self.Tout)
            self.qfixed = np.where(boolArr, self.Qin, self.Qout)
            dTemp = -self.Ct*omega*(self.Ts - self.Tfixed)*self.dt
            self.Ts += dTemp
            self.qs += -self.Cq*omega*(self.qs - self.qfixed)*self.dt
            self.update_R()
        elif relaxation_mode == "average":
            for i in range(len(self.T_average)):
                bin_particles = (self.positions[:,0] > i*self.box_size/len(self.T_average)) & (self.positions[:,0] < (i+1)*self.box_size/len(self.T_average))
                if np.sum(bin_particles) > 0:
                    self.T_average[i] = np.mean(self.Ts[bin_particles])
                    self.q_average[i] = np.mean(self.qs[bin_particles])
                    #update the temperature and humidity of particles in the bin towards the average
                    self.Ts[bin_particles] += -self.Ct*omega*(self.Ts[bin_particles] - self.T_average[i])*self.dt
                    self.qs[bin_particles] += -self.Cq*omega*(self.qs[bin_particles] - self.q_average[i])*self.dt

            self.update_R()
        elif relaxation_mode == "zones":
            #in the boolArr ones indicate inside the cloud, zeros are outside
            boolArr = (self.positions[:,0] > self.box_size/3) & (self.positions[:,0] < 2*self.box_size/3)
            T_average_in = np.average(self.Ts[boolArr])
            T_average_out = np.average(self.Ts[~boolArr])

            q_average_in = np.average(self.qs[boolArr])
            q_average_out = np.average(self.qs[~boolArr])

            


            if mixing_mode == "IECM":
                self.update_q_T()
            elif mixing_mode =="IEM":
                self.qs[boolArr] += -self.Cq*omega*(self.qs[boolArr]-q_average_in)*self.dt
                self.qs[~boolArr] += -self.Cq*omega*(self.qs[~boolArr]-q_average_out)*self.dt
                self.Ts[boolArr] += -self.Ct*omega*(self.Ts[boolArr]-T_average_in)*self.dt
                self.Ts[~boolArr] += -self.Ct*omega*(self.Ts[~boolArr]-T_average_out)*self.dt
            self.update_R()



        else:
            print("Invalid relaxation mode. Use 'zones' , 'fixed' or 'average'.")
        
        return self.Ts, self.qs

    def update_q_T(self, grid_size=128, sigma_cells=2):
        """
        Fast estimation of velocity-domain averages using 2D grid binning and Gaussian blurring.
        
        Parameters:
        -----------
        velocities : np.ndarray
            Shape (nParticles, 2), the velocity vectors.
        q_values : np.ndarray
            Shape (nParticles,), the parameter to average.
        grid_size : int
            Resolution of the velocity grid (e.g., 64x64 or 128x128). Higher means more localized.
        sigma_cells : float
            The standard deviation of the Gaussian kernel, measured in grid cells.
            
        Returns:
        --------
        q_avg_grid : np.ndarray
            A (grid_size, grid_size) map of the averaged q parameter.
        x_edges, y_edges : np.ndarray
            The velocity boundary markers for the grid axes.
        """
        #Calculate everything separately for particles inside and outside the cloud
        boolArr = (self.positions[:,0] > self.box_size/3) & (self.positions[:,0] < 2*self.box_size/3)

        def update_domain(boolArr):
            # 1. Dynamically find velocity bounds based on data spread
            v_min, v_max = self.velocities[boolArr].min(), self.velocities[boolArr].max()
            
            # 2. Map 2D velocity coordinates directly to integer grid indices (O(N) operation)
            x_indices = ((self.velocities[boolArr, 0] - v_min) / (v_max - v_min) * (grid_size - 1)).astype(np.int32)
            y_indices = ((self.velocities[boolArr, 1] - v_min) / (v_max - v_min) * (grid_size - 1)).astype(np.int32)

            # Clip just in case of rounding anomalies at the exact boundary
            np.clip(x_indices, 0, grid_size - 1, out=x_indices)
            np.clip(y_indices, 0, grid_size - 1, out=y_indices)
            
            # 3. Accumulate q values and particle counts into the grid
            q_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
            count_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
            
            np.add.at(q_grid, (x_indices, y_indices), self.qs[boolArr])
            np.add.at(count_grid, (x_indices, y_indices), 1.0)
            
            # 4. Apply Gaussian blur to simulate the Gaussian distance kernel weight (O(Grid) operation)
            # truncate=3.0 limits the kernel radius to 3 standard deviations for speed
            q_blur = gaussian_filter(q_grid, sigma=sigma_cells, mode='wrap', truncate=3.0)
            count_blur = gaussian_filter(count_grid, sigma=sigma_cells, mode='wrap', truncate=3.0)
            
            # 5. Safe division to avoid 0/0 errors in empty velocity regions
            q_avg_grid = np.divide(q_blur, count_blur, out=np.zeros_like(q_blur), where=count_blur > 1e-5)

            target_q = q_avg_grid[x_indices,y_indices]
            
            self.qs[boolArr] += self.Cq*omega*(target_q-self.qs[boolArr])*self.dt


             # 3. Accumulate q values and particle counts into the grid
            T_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
            
            np.add.at(T_grid, (x_indices, y_indices), self.Ts[boolArr])
            
            # 4. Apply Gaussian blur to simulate the Gaussian distance kernel weight (O(Grid) operation)
            # truncate=3.0 limits the kernel radius to 3 standard deviations for speed
            T_blur = gaussian_filter(T_grid, sigma=sigma_cells, mode='wrap', truncate=3.0)
            
            # 5. Safe division to avoid 0/0 errors in empty velocity regions
            T_avg_grid = np.divide(T_blur, count_blur, out=np.zeros_like(T_blur), where=count_blur > 1e-5)

            target_T = T_avg_grid[x_indices,y_indices]
            
            self.Ts[boolArr] += self.Ct*omega*(target_T-self.Ts[boolArr])*self.dt

        update_domain(boolArr)
        update_domain(~boolArr)



    def initialize_R(self):
        self.R = np.zeros_like(self.positions[:,0])
        while True:
            oldR = self.R
            self.update_R()
            if max(abs(self.R-oldR)) < 1e-10:
                break
    
    def update_R(self):
        #Molecular diffusion of water vapor
        #TODO assuming e(T) being saturation vapor pressure is the same as equiPressure
        Kd = Rv*self.Ts/(equiPressure(self.Ts)*Dv)

        #conduction of heat
        Kk = (Lv0/(Rv*self.Ts)-1)*Lv0/(kappa*self.Ts)

        D = 1/(1000*(Kd+Kk))
        # print(np.max(2*D*self.calculate_supersaturation()*dt))
        R_new = self.R**2+2*D*self.calculate_supersaturation()*self.dt
        np.clip(R_new, min=0, out = R_new)
        self.R = np.sqrt(R_new)

        return self.R
    
    def calculate_supersaturation(self):
        #Calculate the supersaturation for each particle
        return superSat(self.p, self.Ts, self.qs)