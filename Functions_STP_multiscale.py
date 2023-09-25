#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#####################################
#Packages utilized in defining functions
#####################################
import math
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import scipy as scp
from scipy import optimize
from scipy.linalg import expm
import scipy.integrate as integrate
from scipy import signal
from scipy.special import erf
from numpy.random import default_rng
rng = default_rng()

#####################################
#Preliminary functions, Stationary solutions, and single simulations
#####################################

def firing(un, theta):
    '''
    Function defining the heaviside firing rates
    Input: Activity profile un and firing threshold
    Output: H(un-theta)
    '''
    x=un-theta
    return np.piecewise(x,[x<0,0<=x],[lambda x: 0,lambda x: 1])

def Tophat(x, low, high):
    '''
    Rectangular function useful in defining synaptic profiles 
    as well as intervals of integration where it is convenient.
    Input: spatial domain x, lower and upper bounds of nonzero interval
    Output: Top-hat function returning 1 for x in [lower,upper] and 0 otherwise
    '''
    return np.piecewise(x,[x<low,(low<=x)&(x<=high),x>high],[lambda x: 0,lambda x: 1,lambda x:0])

def find_interfaces(x, un, theta):
    '''
    The purpose of this function is to find threshold crossings on domain x
    Inputs: spatial domain x, activity profile un, and firing threshold theta
    Output: array of threshold crossing positions
    '''
    y=un-theta #shifts threshold crossings to zero value
    s = np.abs(np.diff(np.sign(y))).astype(bool) #searches for sign changes where True is returned
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1) #returns approximations of interfaces via linear interpolation

def weight(x, Amn, sigmn):
    '''
    The chosen synaptic weight functions.
    Inputs: spatial domain x, amplitude Amn, synaptic footprint sigmn
    Output: the wizard hat weight function chosen in our work.
    '''
    return Amn*np.exp(-np.abs(x)/sigmn)

def stationary_convolution_integral(Aab,sigab,low,upp,x):
    '''
    Function defined specifically for the wizard hat weight function of the
    form Aab*exp(-|x|/sab) where Aab may include plasticity paramters
    Inputs: the weight, gain, lower and upper bounds, 
    spatial vector x
    Output: the convolution integral for stationary bump profiles
    '''
    return np.piecewise(x,[x<low,(low<=x)&(x<=upp),upp<x],
                        [lambda x: -Aab*sigab*np.exp(x/sigab)*(np.exp(-upp/sigab)-np.exp(-low/sigab)),
                         lambda x: Aab*sigab*(2-np.exp((x-upp)/sigab)-np.exp((low-x)/sigab)),
                         lambda x: Aab*sigab*np.exp(-x/sigab)*(np.exp(upp/sigab)-np.exp(low/sigab))])

def broad_stationary_solutions(x,guess,th_e,th_i,
                               Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
                               see=1,sei=2,sie=2,sii=2,
                               tau_e=1,tau_i=1,tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
                               qe0=0,qi0=0,beta_e=0,beta_i=0,
                               alpha_e=0,alpha_i=0,
                               **params):
    '''
    Function for numerically calculating halfwidths under the assumption that both
    E and I populations are active
    Inputs: spatial domain, an initial guess [ae,ai], firing thresholds,
    parameters for weight kernels, timescale parameters, plasticity parameters.
    Output: a dictionary of halfwidths and stationary profiles 
    for plasticity and activity variables
    '''
    #initialize solutions
    ae,ai=np.nan,np.nan
    
    #defining constants for plasticity stationary profiles
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e*(1+QE))
    RI=1/(1+alpha_i*(1+QI))
    
    #defining constants utilized in eqs. 3.3
    Se=2*(RE)*(1+QE)
    Si=2*(RI)*(1+QI)
    
    #These functions return an output used for minimization to find the halfwidths
    #they are based on the piecewise system of equations 3.3
    #Case1 refers to ae>ai. 
    def case1(variables):
        ae,ai=variables
        part1=(Se*Aee*see*np.exp(-ae/see)*np.sinh(ae/see)
               -Si*Aei*sei*np.exp(-ae/sei)*np.sinh(ai/sei))-th_e
        part2=(-Si*Aii*sii*np.exp(-ai/sii)*np.sinh(ai/sii)
               +Se*Aie*sie*(1-np.exp(-ae/sie)*np.cosh(ai/sie)))-th_i
        return ((part1)**2+(part2)**2)
    #Case2 refers to ae<ai
    def case2(variables):
        ae,ai=variables
        part1=(Se*Aee*see*np.exp(-ae/see)*np.sinh(ae/see)
               -Si*Aei*sei*(1-np.exp(-ai/sei)*np.cosh(ae/sei)))-th_e
        part2=(-Si*Aii*sii*np.exp(-ai/sii)*np.sinh(ai/sii)
               +Se*Aie*sie*np.exp(-ai/sie)*np.sinh(ae/sie))-th_i
        return ((part1)**2+(part2)**2)
    
    #These functions check that solutions are valid within our tolerance
    #should result in 0 (or within [0, tolerance])
    def check_case1(ae,ai):
        part1=(Se*Aee*see*np.exp(-ae/see)*np.sinh(ae/see)
               -Si*Aei*sei*np.exp(-ae/sei)*np.sinh(ai/sei))-th_e
        part2=(-Si*Aii*sii*np.exp(-ai/sii)*np.sinh(ai/sii)
               +Se*Aie*sie*(1-np.exp(-ae/sie)*np.cosh(ai/sie)))-th_i
        return np.abs(part1),np.abs(part2)
    def check_case2(ae,ai):
        part1=(Se*Aee*see*np.exp(-ae/see)*np.sinh(ae/see)
               -Si*Aei*sei*(1-np.exp(-ai/sei)*np.cosh(ae/sei)))-th_e
        part2=(-Si*Aii*sii*np.exp(-ai/sii)*np.sinh(ai/sii)
               +Se*Aie*sie*np.exp(-ai/sie)*np.sinh(ae/sie))-th_i
        return np.abs(part1),np.abs(part2)
    
    
    #Solving for the halfwidths in either case
    #Bounds are introduced to avoid RunTime warnings. If the solver diverges well 
    #beyond reasonable expectations, no solution is found and np.nan is returned.
    #tol parameter: tolerance for the gradient for successful termination
    bds=((0,500),(0,500))
    Sltncase1=optimize.minimize(case1, guess, bounds=bds, tol=10**(-16), method='L-BFGS-B') 
    Sltncase2=optimize.minimize(case2, guess, bounds=bds, tol=10**(-16), method='L-BFGS-B')
    
    #what follows are various checks to rule out cases 
    #and check validity of solutions that fit our assumptions
    Tolerance=10**(-6)
    if (Sltncase1.x[0]>Sltncase1.x[1]):
        ae0=Sltncase1.x[0]
        ai0=Sltncase1.x[1]
        check_a,check_b=check_case1(ae0,ai0)
        if (check_a<=Tolerance) and (check_b<=Tolerance):
            ae=ae0
            ai=ai0
    if Sltncase2.x[0]<=Sltncase2.x[1]:
        ae0=Sltncase2.x[0]
        ai0=Sltncase2.x[1]
        check_a,check_b=check_case2(ae0,ai0)
        if (check_a<=Tolerance) and (check_b<=Tolerance):
            ae=ae0
            ai=ai0
    #if no solution then the whole function returns empty arrays
    if np.isnan(ae) or np.isnan(ai) or (ae<0) or (ai<0):
        ae=np.nan
        ai=np.nan
        Result={
        'ae': ae,
        'ai': ai,
        'Ue0': np.empty(len(x)),
        'Ui0': np.empty(len(x)),
        'Qe0': np.empty(len(x)),
        'Qi0': np.empty(len(x)),
        'Re0': np.empty(len(x)),
        'Ri0': np.empty(len(x))
        }
        return Result
    
    #If halfwidths satisfy conditions we generate the bump profiles
    Ue0=(stationary_convolution_integral(Aee*(RE)*(1+QE),see,-ae,ae,x)
        -stationary_convolution_integral(Aei*(RI)*(1+QI),sei,-ai,ai,x))
    Ui0=(stationary_convolution_integral(Aie*(RE)*(1+QE),sie,-ae,ae,x)
        -stationary_convolution_integral(Aii*(RI)*(1+QI),sii,-ai,ai,x))
    
    #Below are tests to check that there is a single active region
    #Violations cause the solution to be thrown out
    CheckE=Ue0-th_e
    CheckI=Ui0-th_i
    if (np.sum(CheckE[1:]*CheckE[:-1]<0)==2) and (np.sum(CheckI[1:]*CheckI[:-1]<0)==2) and (CheckE[int(len(x)/2)]>0) and (CheckI[int(len(x)/2)]>0):
        QE0=Tophat(x,-ae,ae)*QE
        QI0=Tophat(x,-ai,ai)*QI
        RE0=np.ones(len(x))-Tophat(x,-ae,ae)*(1-RE)
        RI0=np.ones(len(x))-Tophat(x,-ai,ai)*(1-RI)
        Result={
        'ae': ae,
        'ai': ai,
        'Ue0': Ue0,
        'Ui0': Ui0,
        'Qe0': QE0,
        'Qi0': QI0,
        'Re0': RE0,
        'Ri0': RI0
        }
    else:
        ae=np.nan
        ai=np.nan
        Result={
        'ae': ae,
        'ai': ai,
        'Ue0': np.empty(len(x)),
        'Ui0': np.empty(len(x)),
        'Qe0': np.empty(len(x)),
        'Qi0': np.empty(len(x)),
        'Re0': np.empty(len(x)),
        'Ri0': np.empty(len(x))
        }
    return Result
        
    
def narrow_stationary_solutions(x,guess,th_e,th_i,
                               Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
                               see=1,sei=2,sie=2,sii=2,
                               tau_e=1,tau_i=1,tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
                               qe0=0,qi0=0,beta_e=0,beta_i=0,
                               alpha_e=0,alpha_i=0,
                               **params):
    '''
    Function for numerically calculating halfwidths under the assumption that only the
    E population is active
    Inputs: spatial domain, an initial guess for ae, firing thresholds,
    parameters for weight kernels, timescale parameters, plasticity parameters.
    Output: a dictionary of halfwidths and stationary profiles 
    for plasticity and activity variables
    '''
    #initialize solutions
    ae,ai=np.nan,0
    
    #defining constants for plasticity stationary profiles
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e*(1+QE))
    RI=1/(1+alpha_i*(1+QI))
    
    #defining constants utilized in eqs. 3.3
    Se=2*(RE)*(1+QE)
    Si=2*(RI)*(1+QI)
    
    #This function returns an output used for minimization to find ae
    #it is based on equation 3.4 
    def case1narrow(variable):
        ae=variable
        part1=Se*Aee*see*np.exp(-ae/see)*np.sinh(ae/see)-th_e
        return (part1)**2
    #This function checks that the solution is valid within our tolerance
    #should result in 0 (or within [0, tolerance])
    def check_case1narrow(ae):
        part1=Se*Aee*see*np.exp(-ae/see)*np.sinh(ae/see)-th_e
        return np.abs(part1)
    
    #Solving for the halfwidths 
    Sltn_narrow=optimize.minimize(case1narrow, guess, tol=10**(-16)) 
    ae0=Sltn_narrow.x[0]
    
    #what follows are various checks to rule out cases 
    #and check validity of solutions that fit our assumptions
    Tolerance=10**(-6)
    if check_case1narrow(ae0)<=Tolerance:
        ae=ae0
        
    if (ae<0) or (np.isnan(ae)):
        ae=np.nan
        ai=np.nan
        Result={
        'ae': ae,
        'ai': ai,
        'Ue0': np.empty(len(x)),
        'Ui0': np.empty(len(x)),
        'Qe0': np.empty(len(x)),
        'Qi0': np.empty(len(x)),
        'Re0': np.empty(len(x)),
        'Ri0': np.empty(len(x))
        }
        return Result
    
    #Profiles are calculated for valid solutions
    Ue0=stationary_convolution_integral(Aee*(RE)*(1+QE),see,-ae,ae,x)
    Ui0=stationary_convolution_integral(Aie*(RE)*(1+QE),sie,-ae,ae,x)
    
    #Below are tests to check that there is a single active region
    #Violations cause the solution to be thrown out
    CheckE=Ue0-th_e
    CheckI=Ui0-th_i
    if (np.sum(CheckE[1:]*CheckE[:-1]<0)==2) and (CheckE[int(len(x)/2)]>0) and (CheckI[int(len(x)/2)]<=0+Tolerance):
        QE0=Tophat(x,-ae,ae)*QE
        QI0=Tophat(x,-ai,ai)*QI
        RE0=np.ones(len(x))-Tophat(x,-ae,ae)*(1-RE)
        RI0=np.ones(len(x))-Tophat(x,-ai,ai)*(1-RI)
        Result={
        'ae': ae,
        'ai': ai,
        'Ue0': Ue0,
        'Ui0': Ui0,
        'Qe0': QE0,
        'Qi0': QI0,
        'Re0': RE0,
        'Ri0': RI0
        }
    else:
        ae=np.nan
        ai=np.nan
        Result={
        'ae': ae,
        'ai': ai,
        'Ue0': np.empty(len(x)),
        'Ui0': np.empty(len(x)),
        'Qe0': np.empty(len(x)),
        'Qi0': np.empty(len(x)),
        'Re0': np.empty(len(x)),
        'Ri0': np.empty(len(x))
        }
    return Result

def single_simulation_allFrames(x,th_e,th_i,eps,dt,nt,
                                Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
                                see=1,sei=2,sie=2,sii=2,
                                tau_e=1,tau_i=1,tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
                                qe0=0,qi0=0,beta_e=0,beta_i=0,
                                alpha_e=0,alpha_i=0,
                                Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
                                **params):
    '''
    Function runs a single simulation and stores/returns full profiles per timestep.
    This function is utilized specifically for generating heatmaps of single bumps over time
    Noise may be added or removed via the eps parameter.
    Inputs: spatial domain x, firing thresholds per population, eps (will be zero if no noise), 
    timestep dt, number of timesteps nt, Synaptic profile parameters, timescale parameters, plasticity parameters, and initial profiles
    '''
    dx=x[5]-x[4]
    lx=len(x)
    
    #Initialize useful constants and profiles. 
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e*(1+QE))
    RI=1/(1+alpha_i*(1+QI))
    Ue,Ui,Qe,Qi,Re,Ri=np.copy(Ue0),np.copy(Ui0),np.copy(Qe0),np.copy(Qi0),np.copy(Re0),np.copy(Ri0)
    
    #define the spatial filter for noise
    kernelnoise=np.exp(-x**2)
    
    #initialize solution matrices
    Ep,Ip=np.zeros((len(x),nt)),np.zeros((len(x),nt))
    QEp,QIp=np.zeros((len(x),nt)),np.zeros((len(x),nt))
    REp,RIp=np.zeros((len(x),nt)),np.zeros((len(x),nt))
    
    #initialize profiles
    Ep[:,0]=Ue0
    Ip[:,0]=Ui0
    QEp[:,0]=Qe0
    QIp[:,0]=Qi0
    REp[:,0]=Re0
    RIp[:,0]=Ri0
    
    #noiseless simulation, utilizes forward Euler method for simulation
    if eps==0:
        for i in range(nt-1):
            Ue+=(dt/tau_e)*(-Ue+(signal.fftconvolve(weight(x,Aee,see), Re*(1+Qe)*firing(Ue,th_e),'same')
                            -signal.fftconvolve(weight(x,Aei,sei), Ri*(1+Qi)*firing(Ui,th_i),'same'))*dx)
            Ui+=(dt/tau_i)*(-Ui+signal.fftconvolve(weight(x,Aie,sie), Re*(1+Qe)*firing(Ue,th_e),'same')*dx
                            -signal.fftconvolve(weight(x,Aii,sii), Ri*(1+Qi)*firing(Ui,th_i),'same')*dx)
            Qe+=dt*(-Qe+beta_e*(qe0-Qe)*firing(Ue,th_e))/tau_qe
            Qi+=dt*(-Qi+beta_i*(qi0-Qi)*firing(Ui,th_i))/tau_qi
            Re+=dt*(1-Re-alpha_e*Re*(1+Qe)*firing(Ue,th_e))/tau_re
            Ri+=dt*(1-Ri-alpha_i*Ri*(1+Qi)*firing(Ui,th_i))/tau_ri

            Ep[:,i+1]=Ue
            Ip[:,i+1]=Ui
            QEp[:,i+1]=Qe
            QIp[:,i+1]=Qi
            REp[:,i+1]=Re
            RIp[:,i+1]=Ri
    else: #noisy simulation, utilizes the Milstein Method for SDEs
        for i in range(nt-1):
            We=signal.fftconvolve(kernelnoise,rng.standard_normal(lx),'same')*dx**0.5
            Wi=signal.fftconvolve(kernelnoise,rng.standard_normal(lx),'same')*dx**0.5
            Ue+=(dt/tau_e)*(-Ue+(signal.fftconvolve(weight(x,Aee,see), Re*(1+Qe)*firing(Ue,th_e),'same')
                            -signal.fftconvolve(weight(x,Aei,sei), Ri*(1+Qi)*firing(Ui,th_i),'same'))*dx)+(np.sqrt(eps*dt*np.abs(Ue))*We+dt*eps*np.sign(Ue)*(We*We-np.ones(len(We)))/4)/tau_e
            Ui+=(dt/tau_i)*(-Ui+signal.fftconvolve(weight(x,Aie,sie), Re*(1+Qe)*firing(Ue,th_e),'same')*dx
                            -signal.fftconvolve(weight(x,Aii,sii), Ri*(1+Qi)*firing(Ui,th_i),'same')*dx)+(np.sqrt(eps*dt*np.abs(Ui))*Wi+dt*eps*np.sign(Ui)*(Wi*Wi-np.ones(len(Wi)))/4)/tau_i
            Qe+=dt*(-Qe+beta_e*(qe0-Qe)*firing(Ue,th_e))/tau_qe
            Qi+=dt*(-Qi+beta_i*(qi0-Qi)*firing(Ui,th_i))/tau_qi
            Re+=dt*(1-Re-alpha_e*Re*(1+Qe)*firing(Ue,th_e))/tau_re
            Ri+=dt*(1-Ri-alpha_i*Ri*(1+Qi)*firing(Ui,th_i))/tau_ri

            Ep[:,i+1]=Ue
            Ip[:,i+1]=Ui
            QEp[:,i+1]=Qe
            QIp[:,i+1]=Qi
            REp[:,i+1]=Re
            RIp[:,i+1]=Ri
    return Ep,Ip,QEp,QIp,REp,RIp
    
#####################################
#Stability classification
#####################################


def Ge(sign,alpha_e,qe0,beta_e):
    '''
    Function defining the perturbation to bump edges dependent on the sign of perturbation.
    Input: perturbation sign, plasticity parameters
    Output: Gn as is defined in equation 3.7
    '''
    QE=qe0*beta_e/(1+beta_e)
    if sign>=0:
        return 1
    else:
        return (1+QE)/(1+alpha_e*(1+QE))
    
def Gi(sign,alpha_i,qi0,beta_i):
    '''
    Function defining the perturbation to bump edges dependent on the sign of perturbation.
    Input: perturbation sign, plasticity parameters
    Output: Gn as is defined in equation 3.7
    '''
    QI=qi0*beta_i/(1+beta_i)
    if sign>=0:
        return 1
    else:
        return (1+QI)/(1+alpha_i*(1+QI))
    
    
def STPstabMatrix(Signs,U_epae,U_ipai,ae,ai,
                  Aee,Aei,Aie,Aii,
                  see,sei,sie,sii,
                  tau_e,tau_i,tau_qe,tau_qi,tau_re,tau_ri,
                  qe0,qi0,beta_e,beta_i,
                  alpha_e,alpha_i):
    '''
    Function to create the coefficient matrix of the system of equations utilized to calculate Eigenvalues/Eigenvectors
    Input: takes in an array of the four signs(+/- 1) of the perturbations, 
    the absolute value of the gradient at the interfaces of each population, 
    halfwidths, synaptic weight profile parameters, timescales, and plasticity parameters
    Output: the stability matrix for the system equations 
    '''
    
    #Defining useful constants
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    
    #Defining each row in the matrix
    r1=np.array([-1+U_epae*weight(0,Aee,see)*(Ge(Signs[0],alpha_e,qe0,beta_e)), 
                 U_epae*weight(2*ae,Aee,see)*(Ge(Signs[1],alpha_e,qe0,beta_e)), 
                 -U_ipai*weight(ae-ai,Aei,sei)*(Gi(Signs[2],alpha_i,qi0,beta_i)),
                 -U_ipai*weight(ae+ai,Aei,sei)*(Gi(Signs[3],alpha_i,qi0,beta_i)),
                 RE,0,-RI,0,0,0,0,0,
                 (1+QE),0,-(1+QI),0,0,0,0,0])*(1/tau_e)
    
    r2=np.array([U_epae*weight(2*ae,Aee,see)*(Ge(Signs[0],alpha_e,qe0,beta_e)),
                 -1+U_epae*weight(0,Aee,see)*(Ge(Signs[1],alpha_e,qe0,beta_e)),
                 -U_ipai*weight(ae+ai,Aei,sei)*(Gi(Signs[2],alpha_i,qi0,beta_i)),
                 -U_ipai*weight(ae-ai,Aei,sei)*(Gi(Signs[3],alpha_i,qi0,beta_i)),
                 0,RE,0,-RI,0,0,0,0,
                 0,(1+QE),0,-(1+QI),0,0,0,0])*(1/tau_e)
    
    r3=np.array([U_epae*weight(ae-ai,Aie,sie)*(Ge(Signs[0],alpha_e,qe0,beta_e)),
                 U_epae*weight(ae+ai,Aie,sie)*(Ge(Signs[1],alpha_e,qe0,beta_e)),
                 -U_ipai*weight(0,Aii,sii)*(Gi(Signs[2],alpha_i,qi0,beta_i))-1,
                 -U_ipai*weight(2*ai,Aii,sii)*(Gi(Signs[3],alpha_i,qi0,beta_i)),
                 0,0,0,0,RE,0,-RI,0,
                 0,0,0,0,(1+QE),0,-(1+QI),0])*(1/tau_i)
    
    r4=np.array([U_epae*weight(ae+ai,Aie,sie)*(Ge(Signs[0],alpha_e,qe0,beta_e)),
                 U_epae*weight(ae-ai,Aie,sie)*(Ge(Signs[1],alpha_e,qe0,beta_e)),
                 -U_ipai*weight(2*ai,Aii,sii)*(Gi(Signs[2],alpha_i,qi0,beta_i)),
                 -U_ipai*weight(0,Aii,sii)*(Gi(Signs[3],alpha_i,qi0,beta_i))-1,
                 0,0,0,0,0,RE,0,-RI,
                 0,0,0,0,0,(1+QE),0,-(1+QI)])*(1/tau_i)
    
    r5=np.array([U_epae*weight(0,Aee,see)*beta_e*qe0*np.heaviside(Signs[0],0),
                 U_epae*beta_e*qe0*weight(2*ae,Aee,see)*np.heaviside(Signs[1],0),
                 0,0,-1-beta_e,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0])*(1/tau_qe)
    
    r6=np.array([U_epae*beta_e*qe0*weight(2*ae,Aee,see)*np.heaviside(Signs[0],0),
                 U_epae*beta_e*qe0*weight(0,Aee,see)*np.heaviside(Signs[1],0),
                 0,0,0,-1-beta_e,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*(1/tau_qe)

    r7=np.array([0,0,U_ipai*beta_i*qi0*weight(ae-ai,Aei,sei)*np.heaviside(Signs[2],0),
                 U_ipai*beta_i*qi0*weight(ae+ai,Aei,sei)*np.heaviside(Signs[3],0),
                 0,0,-1-beta_i,0,0,0,0,0,0,0,0,0,0,0,0,0])*(1/tau_qi)
    
    r8=np.array([0,0,U_ipai*beta_i*qi0*weight(ae+ai,Aei,sei)*np.heaviside(Signs[2],0),
                 U_ipai*beta_i*qi0*weight(ae-ai,Aei,sei)*np.heaviside(Signs[3],0),
                 0,0,0,-1-beta_i,0,0,0,0,0,0,0,0,0,0,0,0])*(1/tau_qi)
    
    r9=np.array([U_epae*beta_e*qe0*weight(ae-ai,Aie,sie)*np.heaviside(Signs[0],0),
                 U_epae*beta_e*qe0*weight(ae+ai,Aie,sie)*np.heaviside(Signs[1],0),
                 0,0,0,0,0,0,-1-beta_e,0,0,0,0,0,0,0,0,0,0,0])*(1/tau_qe)
    
    r10=np.array([U_epae*beta_e*qe0*weight(ae+ai,Aie,sie)*np.heaviside(Signs[0],0),
                  U_epae*beta_e*qe0*weight(ae-ai,Aie,sie)*np.heaviside(Signs[1],0),
                  0,0,0,0,0,0,0,-1-beta_e,0,0,0,0,0,0,0,0,0,0])*(1/tau_qe)
    
    r11=np.array([0,0,U_ipai*weight(0,Aii,sii)*beta_i*qi0*np.heaviside(Signs[2],0),
                  U_ipai*beta_i*qi0*weight(2*ai,Aii,sii)*np.heaviside(Signs[3],0),
                  0,0,0,0,0,0,-1-beta_i,0,0,0,0,0,0,0,0,0])*(1/tau_qi)
    
    r12=np.array([0,0,U_ipai*beta_i*qi0*weight(2*ai,Aii,sii)*np.heaviside(Signs[2],0),
                  U_ipai*beta_i*qi0*weight(0,Aii,sii)*np.heaviside(Signs[3],0),
                  0,0,0,0,0,0,0,-1-beta_i,0,0,0,0,0,0,0,0])*(1/tau_qi)

    r13=np.array([-U_epae*weight(0,Aee,see)*alpha_e*np.heaviside(Signs[0],0),
                 -U_epae*alpha_e*weight(2*ae,Aee,see)*np.heaviside(Signs[1],0),
                 0,0,0,0,0,0,0,0,0,0,-1-alpha_e*(1+QE),0,0,0,0,0,0,0])*(1/tau_re)
    
    r14=np.array([-U_epae*alpha_e*weight(2*ae,Aee,see)*np.heaviside(Signs[0],0),
                   -U_epae*alpha_e*weight(0,Aee,see)*np.heaviside(Signs[1],0),
                   0,0,0,0,0,0,0,0,0,0,0,-1-alpha_e*(1+QE),0,0,0,0,0,0])*(1/tau_re)

    r15=np.array([0,0,-U_ipai*alpha_i*weight(ae-ai,Aei,sei)*np.heaviside(Signs[2],0),
                 -U_ipai*alpha_i*weight(ae+ai,Aei,sei)*np.heaviside(Signs[3],0),
                 0,0,0,0,0,0,0,0,0,0,-1-alpha_i*(1+QI),0,0,0,0,0])*(1/tau_ri)
    
    r16=np.array([0,0,-U_ipai*alpha_i*weight(ae+ai,Aei,sei)*np.heaviside(Signs[2],0),
                 -U_ipai*alpha_i*weight(ae-ai,Aei,sei)*np.heaviside(Signs[3],0),
                  0,0,0,0,0,0,0,0,
                 0,0,0,-1-alpha_i*(1+QI),0,0,0,0])*(1/tau_ri)
    
    r17=np.array([-U_epae*alpha_e*weight(ae-ai,Aie,sie)*np.heaviside(Signs[0],0),
                 -U_epae*alpha_e*weight(ae+ai,Aie,sie)*np.heaviside(Signs[1],0),
                 0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,-1-alpha_e*(1+QE),0,0,0])*(1/tau_re)
    
    r18=np.array([-U_epae*alpha_e*weight(ae+ai,Aie,sie)*np.heaviside(Signs[0],0),
                  -U_epae*alpha_e*weight(ae-ai,Aie,sie)*np.heaviside(Signs[1],0),
                  0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,-1-alpha_e*(1+QE),0,0])*(1/tau_re)
    
    r19=np.array([0,0,-U_ipai*alpha_i*weight(0,Aii,sii)*np.heaviside(Signs[2],0),
                  -U_ipai*alpha_i*weight(2*ai,Aii,sii)*np.heaviside(Signs[3],0),
                  0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,-1-alpha_i*(1+QI),0])*(1/tau_ri)
    
    r20=np.array([0,0,-U_ipai*alpha_i*weight(2*ai,Aii,sii)*np.heaviside(Signs[2],0),
                  -U_ipai*alpha_i*weight(0,Aii,sii)*np.heaviside(Signs[3],0),
                  0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,-1-alpha_i*(1+QI)])*(1/tau_ri)
    
    #compiling rows into the final matrix
    M=np.vstack((r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,
                 r11,r12,r13,r14,r15,r16,r17,r18,r19,r20))
    return M


def Eigenvals(ValVec,Signs,
               Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
               see=1,sei=2,sie=2,sii=2,
               tau_e=1,tau_i=1,tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
               qe0=0,qi0=0,beta_e=0,beta_i=0,
               alpha_e=0,alpha_i=0,
               ae=0,ai=0,Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
               **params):
    '''
    Function defined for calculating eigenvalues from halfwidth branch 1
    Inputs: string indicating if the desired output should be the eigenvalues or eigenvectors,
    array of the perturbations signs at each interface, 
    synaptic profile parameters, plasticity parameters, and initial profiles
    Outputs: eigenvalues or eigenvectors
    '''
    #if the solution DNE it returns an empty vector
    if (np.isnan(ae)) or (np.isnan(ai)):
        DUM=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
             np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        return DUM
    
    #Defining useful constants
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    
    #defines 1 over the absolute value of the gradient at interfaces
    U_epae=1/np.abs(Aee*(RE)*(1+QE)*(1-np.exp(-2*ae/see))+
                Aei*(RI)*(1+QI)*(-np.exp(-np.abs(ai-ae)/sei)+np.exp(-np.abs(ai+ae)/sei)))
    U_ipai=1/np.abs(Aii*(RI)*(1+QI)*(-1+np.exp(-2*ai/sii))+
                Aie*(RE)*(1+QE)*(-np.exp(-np.abs(ai+ae)/sie)+np.exp(-np.abs(ai-ae)/sie)))
    #Creates the matrix
    M=STPstabMatrix(Signs,U_epae,U_ipai,ae,ai,
                    Aee,Aei,Aie,Aii,
                    see,sei,sie,sii,
                    tau_e,tau_i,tau_qe,tau_qi,tau_re,tau_ri,
                    qe0,qi0,beta_e,beta_i,
                    alpha_e,alpha_i)
    #Numerically solves for the eigenvalues/eigenvectors
    eigvals,eigvec=la.eig(M)
    
    if ValVec=='values':
        return eigvals
    if ValVec=='vectors':
        return eigvec
    

def isStable(sol):
    '''
    Input: array of eigenvalues
    Output: -1 if all eigenvalues are stable and 
    +1 if any eigenvalue in unstable (real part is positive above the tolerance)
    Note: a tolerance is added since we expect to have eigenvalues of zero, numerically this can 
    translate to eigenvalues that are positive but very close to zero (ie 2*10^(-16))
    '''
    yn=-1
    for eig in sol:
        if np.real(eig)>10**(-8): #A tolerance is added so as not to incorrectly flag the zero eigenvalue as unstable
            yn=1
    return yn 

def isStable_vec(sol,vec):
    '''
    Input: array of eigenvalues
    Output: -1 if all eigenvalues are stable and 
    +1 if any eigenvalue in unstable (real part is positive above the tolerance)
    Unlike the other isStable function, this one additionally returns 
    the maximal eigenvalue and eigenvector for classification purposes
    '''
    yn=-1
    dum=0
    dumv=0
    for i,eig in enumerate(sol):
        eigvec=vec[:,i]
        if np.real(eig)>10**(-8):
            if np.real(eig)>np.real(dum):
                dum=eig
                dumv=eigvec
            yn=1
    return yn, dum, dumv   

def classifyInstability(eigfunc,x):
    '''
    This function classifies the instability by approximating 
    the even-ness/odd-ness of the eigenfunction
    '''
    dum=abs(sum(eigfunc*np.sign(x)))
    if dum>=100:
        instability='drift'
    elif dum<100:
        instability='oscillatory'
    return instability, dum


def PerturbationProfiles(lamb,Signs,x,eigvec,
               Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
               see=1,sei=2,sie=2,sii=2,
               tau_e=1,tau_i=1,tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
               qe0=0,qi0=0,beta_e=0,beta_i=0,
               alpha_e=0,alpha_i=0,
               ae=0,ai=0,Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
               **params):
    '''
    This function takes in an eigenvalue and eigenvector and returns the associated profile of the perturbation
    Inputs: eigenvalue, perturbation signs at the interfaces, spatial vector x, eigenvector, 
    synaptic weight parameters, timescale parameters, plasticity parameters, and initial profiles
    Outputs: perturbation profiles of the E and I population respectively
    '''
    #Defines convenient constants
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    
    #defines 1 over the absolute value of the gradient at interfaces
    U_epae=1/np.abs(Aee*(RE)*(1+QE)*(1-np.exp(-2*ae/see))+
                Aei*(RI)*(1+QI)*(-np.exp(-np.abs(ai-ae)/sei)+np.exp(-np.abs(ai+ae)/sei)))
    U_ipai=1/np.abs(Aii*(RI)*(1+QI)*(-1+np.exp(-2*ai/sii))+
                Aie*(RE)*(1+QE)*(-np.exp(-np.abs(ai+ae)/sie)+np.exp(-np.abs(ai-ae)/sie)))
    
    
    Mee=U_epae*(beta_e*qe0*(eigvec[0]*weight(x-ae,Aee,see)*np.heaviside(Signs[0],0)+
                eigvec[1]*weight(x+ae,Aee,see)*np.heaviside(Signs[1],0)))/(1+beta_e+tau_qe*lamb)
    Mie=U_epae*(beta_e*qe0*(eigvec[0]*weight(x-ae,Aie,sie)*np.heaviside(Signs[0],0)+
                eigvec[1]*weight(x+ae,Aie,sie)*np.heaviside(Signs[1],0)))/(1+beta_e+tau_qe*lamb)
    Mei=U_ipai*(beta_i*qi0*(eigvec[2]*weight(x-ai,Aei,sei)*np.heaviside(Signs[2],0)+
                eigvec[3]*weight(x+ai,Aei,sei)*np.heaviside(Signs[3],0)))/(1+beta_i+tau_qi*lamb)
    Mii=U_ipai*(beta_i*qi0*(eigvec[2]*weight(x-ai,Aii,sii)*np.heaviside(Signs[2],0)+
                eigvec[3]*weight(x+ai,Aii,sii)*np.heaviside(Signs[3],0)))/(1+beta_i+tau_qi*lamb)

    Nee=U_epae*(-alpha_e*(eigvec[0]*weight(x-ae,Aee,see)*np.heaviside(Signs[0],0)+
                eigvec[1]*weight(x+ae,Aee,see)*np.heaviside(Signs[1],0)))/(1+alpha_e*(1+QE)+tau_re*lamb)
    Nie=U_epae*(-alpha_e*(eigvec[0]*weight(x-ae,Aie,sie)*np.heaviside(Signs[0],0)+
                eigvec[1]*weight(x+ae,Aie,sie)*np.heaviside(Signs[1],0)))/(1+alpha_e*(1+QE)+tau_re*lamb)
    Nei=U_ipai*(-alpha_i*(eigvec[2]*weight(x-ai,Aei,sei)*np.heaviside(Signs[2],0)+
                eigvec[3]*weight(x+ai,Aei,sei)*np.heaviside(Signs[3],0)))/(1+alpha_i*(1+QI)+tau_ri*lamb)
    Nii=U_ipai*(-alpha_i*(eigvec[2]*weight(x-ai,Aii,sii)*np.heaviside(Signs[2],0)+
                eigvec[3]*weight(x+ai,Aii,sii)*np.heaviside(Signs[3],0)))/(1+alpha_i*(1+QI)+tau_ri*lamb)

    Epert=(RE*Mee+(1+QE)*Nee-RI*Mei-(1+QI)*Nei
           +U_epae*(eigvec[0]*weight(x-ae,Aee,see)*Ge(Signs[0],alpha_e,qe0,beta_e)+
                eigvec[1]*weight(x+ae,Aee,see)*Ge(Signs[1],alpha_e,qe0,beta_e))
           -U_ipai*(eigvec[2]*weight(x-ai,Aei,sei)*Gi(Signs[2],alpha_i,qi0,beta_i)+
                eigvec[3]*weight(x+ai,Aei,sei)*Gi(Signs[3],alpha_i,qi0,beta_i)))/(1+tau_e*lamb)

    Ipert=(RE*Mie+(1+QE)*Nie-RI*Mii-(1+QI)*Nii
           +U_epae*(eigvec[0]*weight(x-ae,Aie,sie)*Ge(Signs[0],alpha_e,qe0,beta_e)+
                eigvec[1]*weight(x+ae,Aie,sie)*Ge(Signs[1],alpha_e,qe0,beta_e))
           -U_ipai*(eigvec[2]*weight(x-ai,Aii,sii)*Gi(Signs[2],alpha_i,qi0,beta_i)+
                eigvec[3]*weight(x+ai,Aii,sii)*Gi(Signs[3],alpha_i,qi0,beta_i)))/(1+tau_i*lamb)
    return Epert, Ipert


#####################################
#Variance simulations and predictions
#####################################
def CenterCalculation(x,th_e, th_i, Ue, Ui, Qe, Qi, Re, Ri):
    '''
    A function used to calculate the center of mass of the activity and plasticity profiles.
    Same methods as are described in the paper.
    Inputs: spatial vector, firing thresholds, activity and plasticity profiles
    Output: center of masses for activity and plasticity variables
    '''
    intface_e=find_interfaces(x, Ue, th_e)
    intface_i=find_interfaces(x, Ui, th_i)
    if len(intface_e)>=2:
        cent_e=(max(intface_e)+min(intface_e))/2
    else:
        cent_e=x[np.argmax(Ue)]
    if len(intface_i)>=2:
        cent_i=(max(intface_i)+min(intface_i))/2
    else:
        cent_i=x[np.argmax(Ui)]

    if Qe.sum()!=0:
        cent_qe=np.average(x,weights=Qe)
    else:
        cent_qe=0
    if Qi.sum()!=0:
        cent_qi=np.average(x,weights=Qi)
    else:
        cent_qi=0
    if (1-Re).sum()!=0:
        cent_re=np.average(x,weights=1-Re)
    else:
        cent_re=0
    if (1-Ri).sum()!=0:
        cent_ri=np.average(x,weights=1-Ri)
    else:
        cent_ri=0
    return cent_e, cent_i, cent_qe, cent_qi, cent_re, cent_ri

def SampledCenters_simulation(x,th_e, th_i, eps, dt, sample_vec,
                        Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
                        see=1,sei=2,sie=2,sii=2,
                        tau_e=1, tau_i=1, tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
                        qe0=0,qi0=0,beta_e=0,beta_i=0,
                        alpha_e=0,alpha_i=0,
                        ae=0,ai=0,Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
                        **params):
    '''
    Runs a single simulation of the system and returns the centers of mass sampled at
    particular frames for neural activity and plasticity profiles. Utilizes the Milstein method for SDEs
    Inputs: spatial domain, firing thresholds, noise level, temporal step, an array
    defining the users desired sampling times (the final entry divided by dt yields the end 
    time of the simulation), all remaining parameters (synaptic weight params, timescales,
    plasticity parameters), and initial profiles.
    Output: center of masses for activity and plasticity variables sampled at particular frames in time
    '''
    
    #initializing spatial and temporal parameters
    dx=x[5]-x[4]
    ls=len(sample_vec)
    lx=len(x)
    T=sample_vec[-1]
    nt=int(np.round(T/dt)+1)
    tvec=np.arange(0,T,dt)
    
    #initializing center solution arrays
    cent_e=np.zeros(ls)
    cent_i=np.zeros(ls)
    cent_qe=np.zeros(ls)
    cent_qi=np.zeros(ls)
    cent_re=np.zeros(ls)
    cent_ri=np.zeros(ls)

    #Initializing profiles
    Ue,Ui,Qe,Qi,Re,Ri=np.copy(Ue0),np.copy(Ui0),np.copy(Qe0),np.copy(Qi0),np.copy(Re0),np.copy(Ri0)
    
    #initialize sample counter
    j=0
    
    #Set the spatial filter for noise
    kernelnoise=np.exp(-x**2)
    
    for i in tvec: 
        We=signal.fftconvolve(kernelnoise,rng.standard_normal(lx),'same')*dx**0.5
        Wi=signal.fftconvolve(kernelnoise,rng.standard_normal(lx),'same')*dx**0.5
        
        Ue+=(dt/tau_e)*(-Ue+(signal.fftconvolve(weight(x,Aee,see), Re*(1+Qe)*firing(Ue,th_e),'same')
                        -signal.fftconvolve(weight(x,Aei,sei), Ri*(1+Qi)*firing(Ui,th_i),'same'))*dx)+(np.sqrt(eps*dt*np.abs(Ue))*We+dt*eps*np.sign(Ue)*(We*We-np.ones(len(We)))/4)/tau_e
        Ui+=(dt/tau_i)*(-Ui+signal.fftconvolve(weight(x,Aie,sie), Re*(1+Qe)*firing(Ue,th_e),'same')*dx
                        -signal.fftconvolve(weight(x,Aii,sii), Ri*(1+Qi)*firing(Ui,th_i),'same')*dx)+(np.sqrt(eps*dt*np.abs(Ui))*Wi+dt*eps*np.sign(Ui)*(Wi*Wi-np.ones(len(Wi)))/4)/tau_i
        if qe0!=0:
            Qe+=dt*(-Qe+beta_e*(qe0-Qe)*firing(Ue,th_e))/tau_qe
        if qi0!=0:
            Qi+=dt*(-Qi+beta_i*(qi0-Qi)*firing(Ui,th_i))/tau_qi
        if alpha_e!=0:
            Re+=dt*(1-Re-alpha_e*Re*(1+Qe)*firing(Ue,th_e))/tau_re
        if alpha_i!=0:
            Ri+=dt*(1-Ri-alpha_i*Ri*(1+Qi)*firing(Ui,th_i))/tau_ri
        if j<len(sample_vec):
            if i>=sample_vec[j]:
                cent_e[j],cent_i[j],cent_qe[j],cent_qi[j],cent_re[j],cent_ri[j]=CenterCalculation(x,th_e, th_i, Ue, Ui, Qe, Qi, Re, Ri)
                j+=1
    cent_e[j],cent_i[j],cent_qe[j],cent_qi[j],cent_re[j],cent_ri[j]=CenterCalculation(x,th_e, th_i, Ue, Ui, Qe, Qi, Re, Ri)
    return np.hstack((cent_e,cent_i,cent_qe,cent_qi,cent_re,cent_ri))

def shift(prof, num, fill_value=0):
    '''
    A function used to shift the blurring profiles by the appropriate center shift.
    Input: plasticity profile, the shift (positive is right and negative is left shift), value to fill
    *for the fill value we use zero as we are calculating profiles of QE and RE-1.
    If alterations are made to the code (such as we are shifting QE+1 or RE) then the fill value would need to change.
    Output: shifted profile
    '''
    result = np.empty_like(prof)
    if num > 0:
        result[:num] = fill_value
        result[num:] = prof[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = prof[-num:]
    else:
        result[:] = prof
    return result

def Err(x,x_1,x_2, t, taup, denom):
    '''
    The scaled error function defined for blurred plasticity profiles in eqs 4.6 
    '''
    eep=0.0000001 #eep is a small offset for the denominator to avoid division errors for t=0
    return 0.5*(erf((x_2-x)/(denom*(1-np.exp(-t/taup)+eep)))
                -erf((x_1-x)/(denom*(1-np.exp(-t/taup)+eep))))

def Pnm(g, h):
    '''
    A function returning the sum of two functions multiplied together. 
    It is utilized in calculating integrals in the predictions. 
    '''
    dum=g*h
    #Below dum!=0 skips the zero entries due to the tophat and saves some calculation time
    return sum(dum[dum!=0]) 


def SampledCenter_predictions(x, th_e, th_i, eps, dt, sample_vec,
                                Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
                                see=1,sei=2,sie=2,sii=2,
                                tau_e=1, tau_i=1, tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
                                qe0=0,qi0=0,beta_e=0,beta_i=0,
                                alpha_e=0,alpha_i=0,
                                ae=0,ai=0,Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
                                **params):
    '''
    Simulates the lower dimensional system and is utlized for predicting variance.
    Inputs: spatial domain, firing thresholds, noise level, temporal step, an array
    defining the users desired sampling times (the final entry divided by dt yields the end 
    time of the simulation), all remaining parameters (synaptic weight params, timescales,
    plasticity parameters), and initial profiles ***NOTE: ae, ai, Ue0, and Ui0 should be the initial profiles 
    for the case with No plasticity applied***.
    Output: center of masses for activity and plasticity variables sampled at particular frames in time
    '''
    #defining convenient constants
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    
    #defining spatial/temporal parameters
    dx=x[5]-x[4]
    indzero=int(len(x)/2)
    lx=len(x)
    T=sample_vec[-1]
    tvec=np.arange(0,T,dt)
    
    #initalizing centers
    center_e,center_i=0,0
    center_qe,center_qi,center_re,center_ri=0,0,0,0

    #initializing sample counter
    j=0
    
    #defining spatial filter for noise
    kernelnoise=np.exp(-x**2)

    #initializing solution vectors
    ls=len(sample_vec)
    cent_e=np.zeros(ls)
    cent_i=np.zeros(ls)
    cent_qe=np.zeros(ls)
    cent_qi=np.zeros(ls)
    cent_re=np.zeros(ls)
    cent_ri=np.zeros(ls)
    
    #defining the absolute value of the gradients at the interfaces
    U_epae=np.abs(Aee-weight(2*ae,Aee,see)-weight(ae-ai,Aei,sei)+weight(ae+ai,Aei,sei))
    U_ipai=np.abs(-Aii+weight(2*ai,Aii,sii)-weight(ai+ae,Aie,sie)+weight(ai-ae,Aie,sie))
    
    #defining useful arrays and constants called upon multiple times in the simulation
    ke=weight(ae+ai,Aei,sei)-weight(ae-ai,Aei,sei)
    ki=weight(ae+ai,Aie,sie)-weight(ae-ai,Aie,sie)
    
    ingral1=Tophat(x,-ae,ae)*(weight(ae-x,Aee,see)-weight(-ae-x,Aee,see))
    ingral2=Tophat(x,-ai,ai)*(weight(ae-x,Aei,sei)-weight(-ae-x,Aei,sei))
    ingral3=Tophat(x,-ae,ae)*(weight(ai-x,Aie,sie)-weight(-ai-x,Aie,sie))
    ingral4=Tophat(x,-ai,ai)*(weight(ai-x,Aii,sii)-weight(-ai-x,Aii,sii))

    E0=np.abs(np.copy(Ue0))
    I0=np.abs(np.copy(Ui0))
    
    denom_erf_qe=np.sqrt(2*eps*((E0+I0)*tau_qe))
    denom_erf_qi=np.sqrt(2*eps*((E0+I0)*tau_qi))
    denom_erf_re=np.sqrt(2*eps*((E0+I0)*tau_re))
    denom_erf_ri=np.sqrt(2*eps*((E0+I0)*tau_ri))
    
    #initializing plasticity profiles 
    #NOTE! Rn is fed in as Rn-1, we found it was more convenient to update Rn-1 in these calculations than Rn
    RQe,RQi,Re,Ri=np.copy(Re0)*np.copy(Qe0),np.copy(Ri0)*np.copy(Qi0),np.copy(Re0)-1,np.copy(Ri0)-1
    
    for time in tvec:
        We=signal.fftconvolve(kernelnoise,rng.standard_normal(lx),'same') #filtered noise
        Wi=signal.fftconvolve(kernelnoise,rng.standard_normal(lx),'same') #filtered noise
        Wexe=(We[int((center_e+ae)/dx+indzero)]-We[int((center_e-ae)/dx+indzero)])*dx**0.5
        Wixi=(Wi[int((center_i+ai)/dx+indzero)]-Wi[int((center_i-ai)/dx+indzero)])*dx**0.5

        center_e+=(dt*(-2*ke*(center_e-center_i)
                       +(Pnm(RQe+Re, ingral1)-Pnm(RQi+Ri, ingral2))*dx)
                   +(eps*dt*th_e)**0.5*(Wexe))/(2*tau_e*U_epae)
        center_i+=(dt*(-2*ki*(center_e-center_i)
                       +(Pnm(RQe+Re, ingral3)-Pnm(RQi+Ri, ingral4))*dx)
                   +(eps*dt*th_i)**0.5*(Wixi))/(2*tau_i*U_ipai)
        #For the plasticity evolution: 1. centers are calculated, 
        #2. the blurred profile (centered at 0) is calculated,
        #3. the whole profile is shifted
        #one may change how these calculations are approached, this is the currently chosen method.
        
        if alpha_e!=0:
            center_re+=dt*((1+(1+QE)*alpha_e)*(center_e-center_re))/tau_re
            Re=shift((RE-1)*Err(x, -ae,ae,time,tau_re,denom_erf_re),int((center_re-center_e)/dx),0)
        else:
            center_re=0
        if alpha_i!=0:
            center_ri+=dt*((1+(1+QI)*alpha_i)*(center_i-center_ri))/tau_ri
            Ri=shift((RI-1)*Err(x, -ai,ai,time,tau_ri,denom_erf_ri),int((center_ri-center_i)/dx),0)
        else:
            center_ri=0
        if qe0!=0:
            center_qe+=dt*(1+beta_e)*(center_e-center_qe)/tau_qe
            Qe=shift(QE*Err(x, -ae,ae,time,tau_qe,denom_erf_qe),int((center_qe-center_e)/dx),0)
            if alpha_e!=0:
                RQe=Qe*(Re+1)
            else:
                RQe=Qe
        else:
            center_qe=0
        if qi0!=0:
            center_qi+=dt*(1+beta_i)*(center_i-center_qi)/tau_qi
            Qi=shift(QI*Err(x, -ai,ai,time,tau_qi,denom_erf_qi),int((center_qi-center_i)/dx),0)
            if alpha_i!=0:
                RQi=Qi*(Ri+1)
            else:
                RQi=Qi
        else:
            center_qi=0
            
        if j<=ls:
            if time>=sample_vec[j]:
                cent_e[j],cent_i[j],cent_qe[j],cent_qi[j],cent_re[j],cent_ri[j]=center_e,center_i,center_qe,center_qi,center_re,center_ri
                j+=1
    cent_e[j],cent_i[j],cent_qe[j],cent_qi[j],cent_re[j],cent_ri[j]=center_e,center_i,center_qe,center_qi,center_re,center_ri
    return np.hstack((cent_e,cent_i,cent_qe,cent_qi,cent_re,cent_ri))

#####################################
#Older theory and long time diffusion coefficients
#####################################
#Cross correlations for a guassian spatial filter e^(-x^2).
def Ce(x):
    return np.sqrt(np.pi/2)*np.exp(-0.5*x**2)
def Cc(x):
    return 0
def Ci(x):
    return np.sqrt(np.pi/2)*np.exp(-0.5*x**2)

#Results derived in our prior work for comparison
def OldTheory(th_e, th_i, eps, t,
            Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
            see=1,sei=2,sie=2,sii=2,
            tau_e=1, tau_i=1, tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
            qe0=0,qi0=0,beta_e=0,beta_i=0,
            alpha_e=0,alpha_i=0,
            ae=0,ai=0,Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
            **params):
    '''
    Inputs: firing thresholds, noise amplitude, time, synaptic profile parameters, timescales,
    plasticity parameters, initial profiles (for the case of NO plasticity!)
    Output: predicted variance
    '''
    U_epae=np.abs(Aee-weight(2*ae,Aee,see)-weight(ae-ai,Aei,sei)+weight(ae+ai,Aei,sei))
    U_ipai=np.abs(-Aii+weight(2*ai,Aii,sii)-weight(ai+ae,Aie,sie)+weight(ai-ae,Aie,sie))
    
    De=eps*th_e*(Ce(0)-Ce(2*ae))/(2*(tau_e*U_epae)**2)
    Di=eps*th_i*(Ci(0)-Ci(2*ai))/(2*(tau_i*U_ipai)**2)
    Dc=eps*np.sqrt(th_e*th_i)*(Cc(ae-ai)-Cc(ae+ai))/(2*tau_e*tau_i*U_epae*U_ipai)
    
    Me=(weight(ae-ai,Aei,sei)-weight(ae+ai,Aei,sei))/(tau_e*U_epae)
    Mi=(weight(ae-ai,Aie,sie)-weight(ae+ai,Aie,sie))/(tau_i*U_ipai)
    
    u1=(Di*Me**2-2*Dc*Me*Mi+De*Mi**2)/((Me-Mi)**2)
    u2=-2*(np.exp((Me-Mi)*t)-1)*((Dc-Di)*(Me**2)-(De-Dc)*Me*Mi)/((Mi-Me)**3)
    u3=-Me**2*(np.exp(2*(Me-Mi)*t)-1)*(De+Di-2*De)/(2*(Mi-Me)**3)
    
    v1=(Di*Me**2-2*Dc*Me*Mi+De*Mi**2)/((Me-Mi)**2)
    v2=-2*(np.exp((Me-Mi)*t)-1)*((Dc-De)*Mi**2-(Di-Dc)*Me*Mi)/((Mi-Me)**3)
    v3=-Mi**2*(np.exp(2*(Me-Mi)*t)-1)*(De+Di-2*Dc)/(2*(Mi-Me)**3)
    return [u1*t+u2+u3, v1*t+v2+v3]


def LongTime_dstp(x,x_1,x_2, denom):
    '''
    The time derivative of plasticity profiles (exluding amplitudes) in the long time.
    Input: spatial domain, interfaces, denominator of scaled error function in the long time limit.
    Output: shape of plasticity profiles (the amplitudes are appropriately scaled in the simulation)
    '''
    return (np.exp(-(x_2-x)**2/(denom))-np.exp(-(x_1-x)**2/(denom)))/(np.sqrt(np.pi*denom))

def LongTime_Err(x,x_1,x_2, taup, denom):
    '''
    The scaled error function defined for blurred plasticity profiles in eqs 4.6 
    '''
    return 0.5*(erf((x_2-x)/(denom))-erf((x_1-x)/(denom)))

def Diff_coeff_Plastic(x,eps, th_e, th_i, t,
                Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
                see=1,sei=2,sie=2,sii=2,
                tau_e=1, tau_i=1, tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
                qe0=0,qi0=0,beta_e=0,beta_i=0,
                alpha_e=0,alpha_i=0,
                ae=0,ai=0,Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
                **params):
    '''
    Numerically estimates the diffusion coefficient in the long time limit. 
    Inputs: spatial domain, noise level, firing thresholds, endtime (should be chosen to be quite large),
    all remaining parameters (synaptic weight params, timescales,
    plasticity parameters), and initial profiles.
    Output: estimate of the long time diffusion coefficient
    '''
    #defining convenient constants
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    dx=x[5]-x[4]
    U_epae=np.abs(Aee-weight(2*ae,Aee,see)-weight(ae-ai,Aei,sei)+weight(ae+ai,Aei,sei))
    U_ipai=np.abs(-Aii+weight(2*ai,Aii,sii)-weight(ai+ae,Aie,sie)+weight(ai-ae,Aie,sie))
    
    #defining the longtime standar deviation/denominator used to obtain longtime derivatives of plasticity profiles
    E0=np.abs(np.copy(Ue0))
    I0=np.abs(np.copy(Ui0))
    denom_qe=(2*eps*((E0+I0)*tau_qe))
    denom_qi=(2*eps*((E0+I0)*tau_qi))
    denom_re=(2*eps*((E0+I0)*tau_re))
    denom_ri=(2*eps*((E0+I0)*tau_ri))
    
    denom_erf_qe=np.sqrt(2*eps*((E0+I0)*tau_qe))
    denom_erf_qi=np.sqrt(2*eps*((E0+I0)*tau_qi))
    denom_erf_re=np.sqrt(2*eps*((E0+I0)*tau_re))
    denom_erf_ri=np.sqrt(2*eps*((E0+I0)*tau_ri))
    
    Qe=QE*LongTime_Err(x, -ae,ae,tau_qe,denom_erf_qe)
    Qi=QI*LongTime_Err(x, -ai,ai,tau_qi,denom_erf_qi)
    Re=(RE-1)*LongTime_Err(x, -ae,ae,tau_re,denom_erf_re)+1
    Ri=(RI-1)*LongTime_Err(x, -ai,ai,tau_ri,denom_erf_ri)+1
    
    #Defining more convenient constants and constant arrays
    ke=(weight(ae+ai,Aei,sei)-weight(ae-ai,Aei,sei))
    ki=(weight(ae+ai,Aie,sie)-weight(ae-ai,Aie,sie))
    
    ingral1=Tophat(x,-ae,ae)*(weight(ae-x,Aee,see)-weight(-ae-x,Aee,see))
    ingral2=Tophat(x,-ai,ai)*(weight(ae-x,Aei,sei)-weight(-ae-x,Aei,sei))
    ingral3=Tophat(x,-ae,ae)*(weight(ai-x,Aie,sie)-weight(-ai-x,Aie,sie))
    ingral4=Tophat(x,-ai,ai)*(weight(ai-x,Aii,sii)-weight(-ai-x,Aii,sii))
    
    #all the partial spatial derivatives of plasticity
    d_qe=(Re)*QE*LongTime_dstp(x,-ae,ae, denom_qe)
    d_qi=(Ri)*QI*LongTime_dstp(x,-ai,ai, denom_qi)
    d_re=(1+Qe)*(RE-1)*LongTime_dstp(x,-ae,ae, denom_re)
    d_ri=(1+Qi)*(RI-1)*LongTime_dstp(x,-ai,ai, denom_ri)
    
    #Setting up matrix M
    r1=np.array([-2*ke-(((d_re+d_qe)*ingral1).sum()*dx), 
                 2*ke+((d_ri+d_qi)*ingral2).sum()*dx, 
                 (d_qe*ingral1).sum()*dx, -(d_qi*ingral2).sum()*dx, 
                 (d_re*ingral1).sum()*dx, -(d_ri*ingral2).sum()*dx])/(2*U_epae*tau_e)
    r2=np.array([-2*ki-(((d_re+d_qe)*ingral3).sum()*dx), 
                 2*ki+((d_ri+d_qi)*ingral4).sum()*dx, 
                 (d_qe*ingral3).sum()*dx, -(d_qi*ingral4).sum()*dx, 
                 (d_re*ingral3).sum()*dx, -(d_ri*ingral4).sum()*dx])/(2*U_ipai*tau_i)
    if qe0!=0:
        r3=np.array([(1+beta_e)/tau_qe, 0, -(1+beta_e)/tau_qe, 0, 0, 0])
    else:
        r3=np.array([0, 0, 0, 0, 0, 0])
    if qi0!=0:
        r4=np.array([0,(1+beta_i)/tau_qi, 0, -(1+beta_i)/tau_qi, 0, 0])
    else:
        r4=np.array([0, 0, 0, 0, 0, 0])
    if alpha_e!=0:
        r5=np.array([(1+(1+QE)*alpha_e)/tau_re, 0, 0, 0, -(1+(1+QE)*alpha_e)/tau_re, 0])
    else:
        r5=np.array([0, 0, 0, 0, 0, 0])
    if alpha_i!=0:
        r6=np.array([0,(1+(1+QI)*alpha_i)/tau_ri, 0,0, 0, -(1+(1+QI)*alpha_i)/tau_ri])
    else:
        r6=np.array([0, 0, 0, 0, 0, 0])
    M=np.vstack((r1,r2,r3,r4,r5,r6))
    
    #Setting up matrix D_M
    De=eps*th_e*(Ce(0)-Ce(2*ae))/(2*(tau_e*U_epae)**2)
    Di=eps*th_i*(Ci(0)-Ci(2*ai))/(2*(tau_i*U_ipai)**2)
    Dc=eps*np.sqrt(th_e*th_i)*(Cc(ae-ai)-Cc(ae+ai))/(2*tau_e*tau_i*U_epae*U_ipai)
    
    rr1=np.array([De,Dc,0,0,0,0])
    rr2=np.array([Dc,Di,0,0,0,0])
    rr3=np.array([0,0,0,0,0,0])
    rr4=np.array([0,0,0,0,0,0])
    rr5=np.array([0,0,0,0,0,0])
    rr6=np.array([0,0,0,0,0,0])
    DiffMat=np.vstack((rr1,rr2,rr3,rr4,rr5,rr6))
    
    #numerically approximating eq 4.21
    dt=0.5
    tvec=np.arange(0,t, dt)
    testvar=np.zeros_like(M)
    for s in tvec:
        testvar+=expm(M*(t-s))@DiffMat@expm(M.T*(t-s))
    
    #returning the diagonal of the result
    return (np.diag(testvar))*dt/(t)

