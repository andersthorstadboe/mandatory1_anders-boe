import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = self.L/N
        xi, yj = np.linspace(0,self.L,N+1),np.linspace(0,self.L,N+1)
        self.xij, self.yij = np.meshgrid(xi,yj,indexing='ij',sparse=True) 
        return self.xij, self.yij

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.h**2
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2x = self.D2()
        D2y = self.D2()
        Ix, Iy = sparse.eye(self.N+1), sparse.eye(self.N+1)
        return (sparse.kron(D2x, Iy) + sparse.kron(Ix, D2y)).tolil()

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1,self.N+1), dtype=bool)
        B[1:-1,1:-1] = 0

        return np.where(B.ravel() == 1)[0]

    def meshfunc(self,u):

        return sp.lambdify((x,y),u)(self.xij,self.yij)

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        A = self.laplace()
        bnds = self.get_boundary_indices()
        for i in bnds:
            A[i] = 0
            A[i,i] = 1
        A = A.tocsr()

        b = np.zeros((self.N+1, self.N+1))
        b[:,:] = self.meshfunc(self.f)
        uij = self.meshfunc(self.ue)
        b.ravel()[bnds] = uij.ravel()[bnds]

        return A, b

    def l2_error(self, u):
        """Return l2-error norm"""
        uij = self.meshfunc(self.ue)
        return np.sqrt(self.h*self.h * np.sum((u - uij)**2))
    
    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def LagrangeBasis(self, xj, x=x):
        """Construct Lagrange basis for points in xj

        Parameters
        ----------
        xj : array
            Interpolation points (nodes)
        x : Sympy Symbol

        Returns
        -------
        Lagrange basis as a list of Sympy functions
        """
        from sympy import Mul
        n = len(xj)
        ell = []
        numert = Mul(*[x - xj[i] for i in range(n)])
        for i in range(n):
            numer = numert/(x - xj[i])
            denom = Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
            ell.append(numer/denom)
        return ell
    
    def LagrangeFunc2D(self,u,basisx,basisy):
        N, M = u.shape
        f = 0
        for i in range(N):
            for j in range(M):
                f += basisx[i]*basisy[j]*u[i, j]
        return f
    
    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        deg = 4; deg2 = int(np.ceil(deg/2))
        x_e,y_e = np.where(self.xij == x)[0], np.where(self.yij == y)[1]
        
        if x_e.size != 0 and y_e.size != 0:
            val = self.U[x_e,y_e][0]
        else:
            # Finding closest indices
            idx = np.abs(self.xij - x).argmin()
            idy = np.abs(self.yij - y).argmin()

            idx0 = idx-deg2; idx1 = idx+deg2
            idy0 = idy-deg2; idy1 = idy+deg2
 
            if idx-deg < 0:
                idx0 = 0
                idx1 = idx0 + deg
            if idx+deg > len(self.xij):
                idx1 = len(self.xij)-1
                idx0 = idx1 - deg
            if idy-deg < 0:
                idy0 = 0
                idy1 = idy0 + deg
            if idy+deg > len(self.yij[0,:]):
                idy1 = len(self.yij[0,:]) - 1
                idy0 = idy1 - deg

            xl = np.arange(idx0,idx1+1)
            yl = np.arange(idy0,idy1+1)

            lx,ly = self.LagrangeBasis(self.xij[xl,0],x), self.LagrangeBasis(self.yij[0,yl],y)
            U_in = self.U[xl[0]:xl[-1]+1,yl[0]:yl[-1]+1]

            val = self.LagrangeFunc2D(U_in,lx,ly)

        print('Interpolation value = %g' %( val))

        return val

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    print("Exact value = %g" %(ue.subs({x: 0.52, y: 0.63}).n()))
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3, abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n())
    
    H = sol.h*(14/10)
    print("Exact value = %g" %(ue.subs({x: H, y: 1-H}).n()))
    assert abs(sol.eval(H, 1-H) - ue.subs({x: H, y: 1-H}).n()) < 1e-3

    print("Exact value = %g" %(ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()))
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3, abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n())

if __name__ == '__main__':
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1,ue)
    U = sol(101)

    test_convergence_poisson2d()
    test_interpolation()

    fig,ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    surf = ax[0].plot_surface(sol.xij,sol.yij,U,cmap='bwr')
    ax[0].set_title('U')
    ax[0].set_xlabel('X'); ax[0].set_ylabel('Y')
    Ue = sp.lambdify((x,y),ue)(sol.xij,sol.yij)
    surf1 = ax[1].plot_surface(sol.xij,sol.yij,Ue,cmap='viridis')
    ax[1].set_xlabel('X'); ax[1].set_ylabel('Y')
    ax[1].set_title('U_exact')
    #plt.show()