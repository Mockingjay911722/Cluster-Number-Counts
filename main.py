import sys

import numpy as np
from matplotlib import rc, pyplot as plt
from scipy import integrate
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import special
from getdist import plots, MCSamples

# We firstly define our cosmo as global reference
cosmo = cosmology.setCosmology('planck18')
cosmo.H0 = 70
cosmo.Om0 = 0.3
cosmo.Ob0 = 0.05
cosmo.sigma8 = 0.8
cosmo.ns = 0.96
def calculate_comoving_volume_element(Redshift_1d):
    #We firstly create the array of Redshift_1d
    nz = len(Redshift_1d)
    #We secondly create loop in order to calculate the comoving volume
    cvol_z = np.zeros(nz)
    for iz in range(nz):
        cdist_z = cosmo.comovingDistance(z_min=0., z_max=Redshift_1d[iz], transverse=False)  # cdist_z represents the comoving distance with specific z [Mpc]
        E_z = cosmo.Ez(Redshift_1d[iz])  # no .value
        H_z = cosmo.H0 * E_z
        #  We thirdly calculate the comoving volume element at each redshift
        c = 299792.458
        cvol_z[iz] = c * cdist_z ** 2 / H_z  # Mpc^3

    return cvol_z

def calculate_halo_mass_function (mass_values_1d, Redshift_1d):# how to model massfunction_model
    # Initialize an empty array
    nm = len(mass_values_1d)
    nz = len(Redshift_1d)
    # We use the for loop to arrange the values
    results = np.zeros([nm, nz])
    for iz in range(nz):
        redshift = Redshift_1d[iz]
        # We use the halo mass function
        mfunc = mass_function.massFunction(mass_values_1d, redshift, model='watson13' , mdef='500c', q_in='M', q_out='dndlnM')
        results[:,iz]  = mfunc/mass_values_1d
    return results #results is 2_d



def calculate_selection_function(M500c_1d, Redshift_1d):
    nm = len(M500c_1d); nz = len(Redshift_1d)
    selection_function_2d = np.zeros([nm,nz])
    Mth = 10**(14.8) #Msun/h
    #Mth = 10**(13.)
    scatter_1d = 0.25 + (0.40-0.25) /2 * Redshift_1d
    for iz in range(nz):
        scatter_value = scatter_1d[iz]
        argument = np.log(Mth/M500c_1d)/(scatter_value * np.sqrt(2.))
        selection = 0.5 * special.erfc(argument)
        selection_function_2d[:,iz] = selection

    return (selection_function_2d)

# def bin_distribution(Redshift_1d, dn_dzm, z_edges):
#     n_counts, edges = np.histogram(Redshift_1d, bins = z_edges, weights = dn_dzm)
#     z_bins = 0.5 * (edges[1:] + edges[:-1])
#     return (n_counts,z_bins)

def main ():
    #We firstly give related redshift mass values and solid angle
    mass_values_1d = 10 ** np.linspace(12, 16, 700)
    nz = 1000
    Redshift_1d = 10 ** (np.linspace(np.log10(0.001), np.log10(2), nz))
    solid_angle = 4 * np.pi
    z_edges = 30
    #We secondly calculate dn_dz
    # dn_dz = calculate_comoving_volume_element(Redshift_1d) * calculate_halo_mass_function(mass_values_1d, Redshift_1d) *calculate_selection_function(mass_values_1d, Redshift_1d) * solid_angle
    colv_z = calculate_comoving_volume_element(Redshift_1d)
    mass_fun = calculate_halo_mass_function(mass_values_1d, Redshift_1d)
    selc = calculate_selection_function(mass_values_1d, Redshift_1d)
    dn_dz =  colv_z[None,:] * mass_fun * selc * solid_angle
    print('dn_dz.shape')
    print(dn_dz.shape)
    #We thirdly calculate dn_dzm
    dn_dzm = integrate.trapz(dn_dz, mass_values_1d, axis=0)
    print('dn_dzm.shape')
    print(dn_dzm.shape)
    print('dn_dzm.size')
    print(dn_dzm.size)
    print('dn_dzm.ndim')
    print(dn_dzm.ndim)
    # np.savetxt('dn_dzm', dn_dzm)
    #
    # a = open("dn_dzm", 'r')  # open file in read mode
    #
    # print("the file contains:")
    # b = np.load(a)

    # plt.plot(Redshift_1d,dn_dzm)
    # plt.show()
    # sys.exit()
    #We finally define the distribution and its values
    # bin_distribution(Redshift_1d, dn_dzm, z_edges)
    n, zbin_edges = np.histogram(Redshift_1d,bins=30, weights=dn_dzm)
    print('n.shape')
    print(n.shape)
    print('zbin_edges.shape')
    print(zbin_edges.shape)
    rc('font', size=18)
    plt.figure(figsize=(10, 7))
    plt.title(r'Cluster Number Counts')
    plt.xlabel(r'Redshift $z$'); plt.ylabel(r'Number of Clusters ')
    plt.stairs(n, zbin_edges , linewidth=4, color='firebrick' )
    plt.show()
main()


