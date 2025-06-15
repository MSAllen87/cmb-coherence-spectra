#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Install CLASS from GitHub and build it
get_ipython().system('git clone https://github.com/lesgourg/class_public.git')
get_ipython().run_line_magic('cd', 'class_public')
get_ipython().system('make')


# In[10]:


import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

# Define scalar coherence model
t = np.linspace(0.1, 14, 500)
t0 = 14
alpha = 1.0

z = (t0 / t)**alpha - 1
dt = t[1] - t[0]
chi = cumulative_trapezoid(1 / (1 + z)[::-1], dx=dt, initial=0)[::-1]

# Save background file for CLASS
df = pd.DataFrame({'z': z, 'chi': chi}).sort_values(by='z', ascending=False)
df.to_csv('background_alpha_1.0.dat', sep='\t', index=False, header=False)


# In[11]:


with open("coherence_model.ini", "w") as f:
      f.write("""
          output = tCl,pCl,mPk
              l_max_scalars = 2500
                  use_tabulated_background = yes
                      background_table = background_alpha_1.0.dat
                          output_root = output/coh_alpha_1_0
                              omega_b = 0.022
                                  omega_cdm = 0.12
                                      h = 0.67
                                          A_s = 2.1e-9
                                              n_s = 0.965
                                                  tau_reio = 0.06
                                                      z_pk = 0, 0.5, 1.0
                                                          """)


# In[12]:


get_ipython().system('./class coherence_model.ini')


# In[13]:


get_ipython().system('ls -lt output/')


# In[14]:


from google.colab import files
uploaded = files.upload()


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

# Load Planck TT
planck_tt = np.loadtxt("COM_PowerSpect_CMB-TT-binned_R3.01 (3).txt")
ell_tt_planck = planck_tt[:, 0]
cl_tt_planck = planck_tt[:, 1]
cl_tt_err = planck_tt[:, 2]

# Load CLASS TT output
cl = np.loadtxt("output/coherence_model00_cl.dat")
ell = cl[:, 0]
cl_tt = cl[:, 1]

# Normalize CLASS TT to Planck peak
scale_tt = max(cl_tt_planck) / max(cl_tt)
cl_tt_scaled = cl_tt * scale_tt

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(ell_tt_planck, cl_tt_planck, yerr=cl_tt_err, fmt='o', color='black', label="Planck 2018 TT")
plt.plot(ell, cl_tt_scaled, color='royalblue', label="Coherence Model TT")
plt.xlabel(r"Multipole moment $\ell$")
plt.ylabel(r"$C_\ell^{TT}$ [$\mu$K$^2$]")
plt.title("CMB Temperature Power Spectrum (TT)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[17]:


# Load Planck EE
planck_ee = np.loadtxt("COM_PowerSpect_CMB-EE-binned_R3.02.txt")
ell_ee_planck = planck_ee[:, 0]
cl_ee_planck = planck_ee[:, 1]
cl_ee_err = planck_ee[:, 2]

# CLASS EE from same cl file (col index 2)
cl_ee = cl[:, 2]
scale_ee = max(cl_ee_planck) / max(cl_ee)
cl_ee_scaled = cl_ee * scale_ee

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(ell_ee_planck, cl_ee_planck, yerr=cl_ee_err, fmt='o', color='black', label="Planck 2018 EE")
plt.plot(ell, cl_ee_scaled, color='darkgreen', label="Coherence Model EE")
plt.xlabel(r"Multipole moment $\ell$")
plt.ylabel(r"$C_\ell^{EE}$ [$\mu$K$^2$]")
plt.title("E-mode Polarization Spectrum (EE)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[18]:


# Load Planck TE
planck_te = np.loadtxt("COM_PowerSpect_CMB-TE-binned_R3.02.txt")
ell_te_planck = planck_te[:, 0]
cl_te_planck = planck_te[:, 1]
cl_te_err = planck_te[:, 2]

# CLASS TE (column index 3)
cl_te = cl[:, 3]
scale_te = max(abs(cl_te_planck)) / max(abs(cl_te))
cl_te_scaled = cl_te * scale_te

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(ell_te_planck, cl_te_planck, yerr=cl_te_err, fmt='o', color='black', label="Planck 2018 TE")
plt.plot(ell, cl_te_scaled, color='darkred', label="Coherence Model TE")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel(r"Multipole moment $\ell$")
plt.ylabel(r"$C_\ell^{TE}$ [$\mu$K$^2$]")
plt.title("Temperature–Polarization Cross Spectrum (TE)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# Load Planck EE and TE
ee = np.loadtxt("COM_PowerSpect_CMB-EE-binned_R3.02.txt")
te = np.loadtxt("COM_PowerSpect_CMB-TE-binned_R3.02.txt")

ell_ee = ee[:, 0]
cl_ee_planck = ee[:, 1]
cl_ee_err = ee[:, 2]

ell_te = te[:, 0]
cl_te_planck = te[:, 1]
cl_te_err = te[:, 2]

# Load Coherence model from CLASS output
cl = np.loadtxt("output/coherence_model00_cl.dat")
ell_model = cl[:, 0]
cl_ee_model = cl[:, 2]
cl_te_model = cl[:, 3]

# Normalize model to match Planck amplitude visually
scale_ee = max(cl_ee_planck) / max(cl_ee_model)
scale_te = max(abs(cl_te_planck)) / max(abs(cl_te_model))

cl_ee_scaled = cl_ee_model * scale_ee
cl_te_scaled = cl_te_model * scale_te

# Plot: Combined EE and TE on log scale
plt.figure(figsize=(10, 6))
plt.yscale("log")

# Scalar coherence model curves
plt.plot(ell_model, cl_ee_scaled, label="Scalar Coherence EE", color="green", linewidth=1.5)
plt.plot(ell_model, np.abs(cl_te_scaled), label="Scalar Coherence TE", color="orange", linewidth=1.5)

# Planck 2018 binned data with error bars
plt.errorbar(ell_ee, cl_ee_planck, yerr=cl_ee_err, fmt='o', markersize=4, color='green', alpha=0.6, label="Planck EE (binned)")
plt.errorbar(ell_te, np.abs(cl_te_planck), yerr=cl_te_err, fmt='o', markersize=4, color='orange', alpha=0.6, label="Planck TE (binned)")

plt.xlabel("Multipole moment $\ell$")
plt.ylabel(r"$C_\ell\ [\mu K^2]$")
plt.title("Scalar Coherence vs. Planck Polarization")
plt.legend(loc="upper right")
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# In[21]:


from google.colab import files
uploaded = files.upload()


# In[23]:


with open("planck2018_lcdm.ini", "w") as f:
      f.write("""
      # Planck 2018 Best-Fit Parameters - Linear Matter Power Spectrum

      output = mPk
      z_pk = 0,0.5,1.0
      P_k_max_h/Mpc = 1.0

      # Cosmological Parameters
      omega_b = 0.022383
      omega_cdm = 0.120107
      h = 0.67810
      n_s = 0.96605
      A_s = 2.106e-9
      tau_reio = 0.05430842

      # Output control
      l_max_scalars = 2500
      output_root = output/planck2018_lcdm
      """)


# In[24]:


get_ipython().system('./class planck2018_lcdm.ini')


# In[26]:


get_ipython().system('ls -lt output/')


# In[27]:


import numpy as np
import matplotlib.pyplot as plt

# Load Coherence Model P(k)
coh = np.loadtxt("output/coherence_model00_z3_pk.dat")
k_coh = coh[:, 0]
pk_coh_z0 = coh[:, 1]

# Load Planck ΛCDM P(k)
lcdm = np.loadtxt("output/planck2018_lcdm00_z3_pk.dat")
k_lcdm = lcdm[:, 0]
pk_lcdm_z0 = lcdm[:, 1]

# Optional: Normalize scalar coherence to match Planck peak
scale_pk = max(pk_lcdm_z0) / max(pk_coh_z0)
pk_coh_z0_scaled = pk_coh_z0 * scale_pk

# Plot
plt.figure(figsize=(8, 6))
plt.loglog(k_lcdm, pk_lcdm_z0, label="Planck ΛCDM $P(k)$", color='black')
plt.loglog(k_coh, pk_coh_z0_scaled, label="Scalar Coherence $P(k)$", color='blue', linestyle='--')
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$P(k)$ [(Mpc/h)$^3$]")
plt.title("Matter Power Spectrum Comparison at $z = 0$")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

