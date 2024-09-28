import os.path as path
import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        L = 1
        self.N = N
        self.h = L/N
        xi, yj = np.linspace(0,L,N+1),np.linspace(0,L,N+1)
        self.xij, self.yij = np.meshgrid(xi,yj,indexing='ij',sparse=sparse) 
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.h**2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx, ky = self.mx*np.pi, self.my*np.pi
        return self.c*np.sqrt(kx**2 + ky**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.Unp1,self.Un,self.Unm1 = np.zeros((3,N+1,N+1))
        self.Unm1[:] = sp.lambdify((x,y,t), self.ue(mx,my))(self.xij,self.yij,0)
        self.Un[:] = sp.lambdify((x,y,t), self.ue(mx,my))(self.xij,self.yij,self.dt) #(self.Unm1[:] + .5 * (self.c*self.dt)**2*(self.D @ self.Un + self.Un @ self.D.T)#(self.D @ self.Unm1 + self.Unm1 @ self.D.T)
        return self.Unp1, self.Un, self.Unm1

    @property
    def dt(self):
        """Return the time step"""
        return (self.cfl*self.h)/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        uij = sp.lambdify((x,y,t), self.ue(self.mx,self.my))(self.xij,self.yij,t0)
        return np.sqrt(self.h*self.h * np.sum((u - uij)**2))

    def apply_bcs(self):
        """Applying Dirichlet boundary conditions with arbitrary RHS"""
        # x = 0
        self.Unp1[0] = self.ue(self.mx,self.my).subs({x: 0})
        # x = Lx = 1
        self.Unp1[-1] = self.ue(self.mx,self.my).subs({x: 1})
        # y = 0
        self.Unp1[:,0] = self.ue(self.mx,self.my).subs({y: 0})
        # y = Ly = 1
        self.Unp1[:,-1] = self.ue(self.mx,self.my).subs({y: 1})

        return self.Unp1

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        # Mesh and initialization
        self.cfl = cfl; self.c = c
        self.Nt = Nt; self.mx = mx; self.my = my
        self.create_mesh(N,sparse=True)
        self.D = self.D2(N)
        
        self.initialize(N,mx,my)
        l2_err = []

        plotdata = {0: self.Unm1.copy()}
        if store_data == 1: 
            plotdata[1] = self.Un.copy()
        for n in range(Nt):
            self.Unp1[:] = 2*self.Un - self.Unm1 + (self.c*self.dt)**2 * (self.D @ self.Un + self.Un @ self.D.T)
            #Boundary condictions
            self.apply_bcs()  
            # Updating Un, Unm1
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1

            if store_data > 0 and n % store_data == 0: # Storing the Un^th time step data
                plotdata[n] = self.Unm1.copy()
            if store_data == -1:
                l2_err.append(self.l2_error(self.Unm1,n))
        if store_data == -1:
            return self.h, l2_err
        else:
            return plotdata

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)
    
    def animationPlot(self,data,f_name,clrmap=cm.coolwarm):
        """Plotting the data from simulation of wave equation using a surface plot"""
        ## Getting directory where image will be stored
        curr_dir = path.dirname(__file__)
        subfold = 'report'
        new_dir = path.join(curr_dir,subfold)

        #print(data.items())
        fig,ax = plt.subplots(subplot_kw={'projection': '3d'})
        frames = []
        for n, val in data.items():
            surf = ax.plot_surface(self.xij,self.yij,val, rstride=2, cstride=2,cmap=clrmap)
            frames.append([surf])
        anima = anim.ArtistAnimation(fig,frames, interval=400, blit=True,repeat_delay=1000)
        anima.save(path.join(new_dir, f_name),writer='pillow',fps=15)

        #plt.show()


        return 0

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.h**2
        return D

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print(abs(r[-1]-2))
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol_D = Wave2D()
    sol_N = Wave2D_Neumann()
    

    assert 0

if __name__ == '__main__':
    test_convergence_wave2d()

    ## Animation
    N, Nt = 50, 75
    solD = Wave2D()
    data = solD(N,Nt,store_data=1)
    f_name = 'testwave_'+str(N)+'_'+str(Nt)+'.gif'
    solD.animationPlot(data,f_name=f_name,clrmap='viridis')
    
