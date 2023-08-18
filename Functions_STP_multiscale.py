#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings


import math
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import scipy as scp
from scipy import optimize
from cmath import sqrt
from scipy.linalg import expm
import scipy.integrate as integrate
from scipy import signal
from scipy.special import erf
from numpy.random import default_rng
rng = default_rng()

#Progress Bars
from tqdm.notebook import tqdm

def firing(un, theta):
    '''
    Function defining the heaviside firing rates
    Input: Activity profile UN and firing threshold
    Output: H(UN-theta)
    '''
    x=un-theta
    return np.piecewise(x,[x<0,0<=x],[lambda x: 0,lambda x: 1])

def Tophat(x, low, high):
    '''
    Rectangular function useful in defining synaptic profiles.
    Input: spatial domain x, lower and upper bounds of nonzero interval
    '''
    return np.piecewise(x,[x<low,(low<=x)&(x<=high),x>high],[lambda x: 0,lambda x: 1,lambda x:0])

def find_interfaces(x, un, theta):
    '''
    The purpose of this function is to find threshold crossings on domain x
    Inputs: spatial domain x, activity profile UN, and firing threshold theta
    Output: array of threshold crossing positions
    '''
    y=un-theta
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

def weight(x, Amn, sigmn):
    '''
    The chosen synaptic weight functions.
    Inputs: spatial domain x, amplitude Amn, synaptic footprint sigmn
    Output: the wizard hat weight function.
    '''
    return Amn*np.exp(-np.abs(x)/sigmn)

def stationary_convolution_integral(Aab,sigab,low,upp,x):
    '''
    Function defined specifically for the weight function of the
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

    #Solving for the halfwidths in either case.  
    #tol parameter: tolerance for the gradient for successful termination
    #Bounds are included to prevent potential overflow errors. 
    #In cases where the halfwidths diverge well beyond reasonable expectations, 
    #valid solutions will not be found.
    bnds = ((0, 500), (0, 500))
    Sltncase1=optimize.minimize(case1, guess, bounds=bnds, tol=10**(-16), method='L-BFGS-B').x
    Sltncase2=optimize.minimize(case2, guess, bounds=bnds, tol=10**(-16), method='L-BFGS-B').x

    #what follows are various checks to rule out cases 
    #and check validity of solutions
    #that fit our assumptions
    
    Tolerance=10**(-6)
    if (Sltncase1[0]>Sltncase1[1]):
        ae0=Sltncase1[0]
        ai0=Sltncase1[1]
        check_a,check_b=check_case1(ae0,ai0)
        if (check_a<=Tolerance) and (check_b<=Tolerance):
            ae=ae0
            ai=ai0
    if Sltncase2[0]<=Sltncase2[1]:
        ae0=Sltncase2[0]
        ai0=Sltncase2[1]
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
    Inputs: spatial domain, an initial guess [ae,ai], firing thresholds,
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
    #and check validity of solutions
    #that fit our assumptions
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
    Noise may be added or removed
    '''
    dx=x[5]-x[4]
    
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    RE=1/(1+alpha_e*(1+QE))
    RI=1/(1+alpha_i*(1+QI))
    
    Ue,Ui,Qe,Qi,Re,Ri=np.copy(Ue0),np.copy(Ui0),np.copy(Qe0),np.copy(Qi0),np.copy(Re0),np.copy(Ri0)
    
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
    #noiseless simulation
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
    else: #noisy simulation
        for i in range(nt-1):
            We=signal.fftconvolve(np.exp(-x**2),rng.standard_normal(len(x)),'same')*dx**0.5
            Wi=signal.fftconvolve(np.exp(-x**2),rng.standard_normal(len(x)),'same')*dx**0.5
            Ue+=(dt/tau_e)*(-Ue+(signal.fftconvolve(weight(x,Aee,see), Re*(1+Qe)*firing(Ue,th_e),'same')
                            -signal.fftconvolve(weight(x,Aei,sei), Ri*(1+Qi)*firing(Ui,th_i),'same'))*dx)+np.sqrt(eps*dt*np.abs(Ue))*We/tau_e
            Ui+=(dt/tau_i)*(-Ui+signal.fftconvolve(weight(x,Aie,sie), Re*(1+Qe)*firing(Ue,th_e),'same')*dx
                            -signal.fftconvolve(weight(x,Aii,sii), Ri*(1+Qi)*firing(Ui,th_i),'same')*dx)+np.sqrt(eps*dt*np.abs(Ui))*Wi/tau_i
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


def Ge(sign,alpha_e=0,qe0=0,beta_e=0,**params):
    '''
    Function defining the perturbation to bump edges dependent on the sign of perturbation.
    Input: perturbation sign, plasticity parameters
    Output: Gn as is utilized in equation 3.7
    '''
    QE=qe0*beta_e/(1+beta_e)
    if sign>=0:
        return 1
    else:
        return (1+QE)/(1+alpha_e*(1+QE))
    
def Gi(sign,alpha_i=0,qi0=0,beta_i=0,**params):
    '''
    Function defining the perturbation to bump edges dependent on the sign of perturbation.
    Input: perturbation sign, plasticity parameters
    Output: Gn as is utilized in equation 3.7
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
    takes in an array of the four signs(+/- 1) of the perturbations 
    and returns the stability matrix for the system equations 
    '''
    
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    
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
    #######Depression ones below
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
    Inputs: Fixed parameters, plasticity parameters, and initial profiles
    Outputs: eigenvalues or eigenvectors dependent on the argument for valvec
    '''
    #if the solution DNE it returns an empty vector
    if (np.isnan(ae)) or (np.isnan(ai)):
        DUM=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
             np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        return DUM
    
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    
    #defines 1 over the slope at interfaces
    U_epae=1/np.abs(Aee*(RE)*(1+QE)*(1-np.exp(-2*ae/see))+
                Aei*(RI)*(1+QI)*(-np.exp(-np.abs(ai-ae)/sei)+np.exp(-np.abs(ai+ae)/sei)))
    U_ipai=1/np.abs(Aii*(RI)*(1+QI)*(-1+np.exp(-2*ai/sii))+
                Aie*(RE)*(1+QE)*(-np.exp(-np.abs(ai+ae)/sie)+np.exp(-np.abs(ai-ae)/sie)))
    
    M=STPstabMatrix(Signs,U_epae,U_ipai,ae,ai,
                    Aee,Aei,Aie,Aii,
                    see,sei,sie,sii,
                    tau_e,tau_i,tau_qe,tau_qi,tau_re,tau_ri,
                    qe0,qi0,beta_e,beta_i,
                    alpha_e,alpha_i)
    eigvals,eigvec=la.eig(M)
    if ValVec=='values':
        return eigvals
    if ValVec=='vectors':
        return eigvec
    

def isStable(sol):
    '''
    this function returns stability masks for all eigenvalues in sol matrix
    note all eigenvalue vectors should be vstacked
    Returns 1 for unstable eigenvalue and -1 for a stable eigenvalue
    '''
    yn=-1
    for eig in sol:
        if np.real(eig)>10**(-14):
            yn=1
    return yn 

def isStable_vec(sol,vec):
    '''this function returns stability masks for all eigenvalues in sol matrix
    note all eigenvalue vectors should be vstacked. 
    Unlike the other isStable function, this one additionally returns 
    the maximal eigenvalue and eigenvector for classification'''
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
    This function classifies the instability by roughly approximating 
    the even-ness/odd-ness of the eigenfunction
    '''
    dum=abs(sum(eigfunc*np.sign(x)))
    if dum>=100:
        instability='drift'
    elif dum<100:
        instability='oscillatory'
    return instability, dum

def EigenfunctionProfiles(lamb,Signs,x,eigvec,
               Aee=0.5,Aei=0.15,Aie=0.15,Aii=0.01,
               see=1,sei=2,sie=2,sii=2,
               tau_e=1,tau_i=1,tau_qe=1,tau_qi=1,tau_re=1,tau_ri=1,
               qe0=0,qi0=0,beta_e=0,beta_i=0,
               alpha_e=0,alpha_i=0,
               ae=0,ai=0,Ue0=0,Ui0=0,Qe0=0,Qi0=0,Re0=0,Ri0=0,
               **params):
    QE=qe0*beta_e/(1+beta_e)
    QI=qi0*beta_i/(1+beta_i)
    
    RE=1/(1+alpha_e+alpha_e*QE)
    RI=1/(1+alpha_i+alpha_i*QI)
    
    #defines 1 over the slope at interfaces
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

