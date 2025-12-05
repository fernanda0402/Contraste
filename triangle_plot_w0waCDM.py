import numpy as np
import matplotlib.pyplot as plt
import corner
from chainconsumer import ChainConsumer

# Parâmetros para facilitar a leitura
arquivo = "/home/usuario/Documentos/Códigos/Estudo_do_contraste/MCMC/Resultados_w0waCDM_Fulldata.txt"  # Substitua pelo nome do seu arquivo
nomes_dos_parametros = [r"$H_0$", r"$\Omega_{m,0}$", r"$\omega_0$", r"$\omega_a$", r"$\sigma_{8,0}$", r"$M_B$"]  # Nomeie os parâmetros conforme necessário

# Carrega os dados do arquivo .txt
dados = np.loadtxt(arquivo)

# Gera o corner plot usando corner
#fig_corner = corner.corner(dados, labels=nomes_dos_parametros, show_titles=True)
#plt.show()

# Gera o triangle plot usando ChainConsumer
c = ChainConsumer()
c.add_chain(dados, parameters=nomes_dos_parametros)
c.configure(shade_alpha=1, summary=True, colors=["blue"], max_ticks=4,legend_artists=True)
fig_chainconsumer = c.plotter.plot()
fig_chainconsumer.set_size_inches(3 + fig_chainconsumer.get_size_inches())
plt.savefig("MCMC_w0waCDM_completo.pdf", dpi=520, format='pdf', bbox_inches='tight')
plt.show()
