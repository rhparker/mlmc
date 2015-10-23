import numpy as np
import matplotlib.pyplot as plt
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file name')
args = parser.parse_args()
if args.input:
	filename = args.input+".npz"
else:
	filename = "data.npz"

data = np.load(filename)
del1 = data['del1']
del2 = data['del2']
var1 = data['var1']
var2 = data['var2']
chk1 = data['chk1']
kur1 = data['kur1']
cost_data = data['cost_data']
level_data = data['level_data']
eps = cost_data.transpose()[0]
mlmc_cost = cost_data.transpose()[1]
std_cost = cost_data.transpose()[2]

num_levels = len(var1)
levels = np.arange(0, num_levels)

plt.figure(1)
plt.subplot(321)
plt.plot(levels[1:], np.log2(var1)[1:], '--k*', label="P_l - P_l-1" )
plt.plot(levels, np.log2(var2), '-k*', label="P_l" )
plt.xlabel('level l')
plt.ylabel('log2 variance')
plt.legend(loc=3)

plt.subplot(322)
plt.plot(levels[1:], np.log2(del1)[1:], '--k*', label="P_l - P_l-1" )
plt.plot(levels, np.log2(del2), '-k*', label="P_l" )
plt.xlabel('level l')
plt.ylabel('log2 |mean|')
plt.legend(loc=3)

plt.subplot(323)
plt.plot(levels[1:], chk1[1:], '--k*')
plt.xlabel('level l')
plt.ylabel('consistency check')

plt.subplot(324)
plt.plot(levels[1:], kur1[1:], '--k*')
plt.xlabel('level l')
plt.ylabel('kurtosis')

plt.subplot(325)
markers = ['*', 'o', 's', 'x', 'd', '+']
for i, ep in enumerate(eps):
	Nl = np.array( filter(None, level_data[i] ) )
	plt.semilogy( np.arange(0, len(Nl) ), Nl, '--k'+markers[i], label=str(ep) )
plt.xlabel('level l')
plt.ylabel('Nl')
plt.legend()

plt.subplot(326)
plt.loglog(eps, eps**2 * mlmc_cost, '--k*', label="MLMC")
plt.loglog(eps, eps**2 * std_cost, '-k*', label="Std MC")
plt.xlabel('accuracy epsilon')
plt.ylabel('epsilon^2 * cost')
plt.legend()

figure = plt.gcf() 					# get current figure
plt.rc('font', size=6)
figure.set_size_inches(8, 10)
plt.savefig("myplot.png", dpi = 300)

plt.savefig(args.input+".pdf")

# plt.show()