import os
from datetime import datetime as dt
import random
import hashlib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import sympy
import colourPals as cp
from importlib import reload

reload(cp)

PRNG_COLOURS = {
    'prngBBS'    : ['#9ecae1', '#6baed6', '#3182bd', '#08519c'],
    'prngHash'   : ['#fdae6b', '#fd8d3c', '#e6550d', '#a63603'],
    'prngRSA'    : ['#a1d99b', '#74c476', '#31a354', '#006d2c'],
    'prngControl': ['#fc9272', '#fb6a4a', '#de2d26', '#a50f15'],
}


def prngBBS(N, KEY_BITLENGTH, para):
    """
    :param N: number of random number to generate
    :param KEY_BITLENGTH: bit-length of outputs
    :param para: series containing parameters seed, n, phi, p, q, e, d, k
    :return:
    """
    bitSeq = ""
    randomNumList = []

    x = para['seed']
    for i in range(KEY_BITLENGTH*N):
        # Take least significant bit of x
        bitSeq += str(x%2)

        # Print progress...
        if i%100_000 == 0:
            print(f"{i} of {KEY_BITLENGTH*N}\nseed: {para['seed']}\n'bitSeq': {bitSeq}")

        x = pow(x, 2, int(para['n']))
        if len(bitSeq) >= KEY_BITLENGTH:
            # If bitSeq is equal to or more than required key-length, output keys and flush the bitSeq
            randomNumList.append(int(bitSeq, 2))
            if len(randomNumList) == N:
                return randomNumList
            bitSeq = ""


# Seed-length must be >= 64
def prngHash(N, KEY_BITLENGTH, para):
    """
    :param N: number of random number to generate
    :param KEY_BITLENGTH: bit-length of outputs
    :param para: series containing parameters seed, n, phi, p, q, e, d, k
    :return:
    """
    assert KEY_BITLENGTH%4 == 0
    keyLen_hex = KEY_BITLENGTH//4
    hexSeq = ""
    randomNumList = []

    m = math.ceil(N*KEY_BITLENGTH/256)
    x = para['seed']
    for i in range(m):
        # hexlength is odd, pad the front with a '0', otherwise, convert to bytes without padding
        if len(hex(x)[2:])%2 == 0:
            x_bytes = bytes.fromhex(hex(x)[2:])
        else:
            x_bytes = bytes.fromhex('0' + hex(x)[2:])

        hexSeq += hashlib.sha256(x_bytes).hexdigest()
        # hexSeq += hashlib.blake2b(x_bytes).hexdigest()

        # Print progress...
        if i%10_000 == 0:
            print(f"{i} of {m}\nseed: {para['seed']}\nhexSeq: {hexSeq}")

        if len(hexSeq) >= keyLen_hex:
            # If hexSeq is equal to or more than required key-length, output keys and flush the hexSeq
            for j in range(0, len(hexSeq), keyLen_hex):
                randomNumList.append(int(hexSeq[j: j + keyLen_hex], 16))
                # print(int(hexSeq[j: j + keyLen_hex], 16))
                # If N random numbers are generated, return list of numbers
                if len(randomNumList) == N:
                    return randomNumList
            hexSeq = ""

        x = (x + 1)%pow(2, para['seedLen'])


# Seed-length must be >= 2*strength where strength = {112, 128, 192, 256}, tf min seed length is 224
def prngRSA(N, KEY_BITLENGTH, para):
    """
    :param N: number of random number to generate
    :param KEY_BITLENGTH: bit-length of outputs
    :param para: series containing parameters seed, n, phi, p, q, e, d, k
    :return:
    """
    seedLen_hex = len(hex(para['seed'])[2:])
    keyLen_hex = KEY_BITLENGTH//4
    k_hexlength = para['k']//4
    hexSeq = ""
    randomNumList = []

    m = math.ceil(N*KEY_BITLENGTH/para['k'])
    x = para['seed']
    for i in range(m):
        y = pow(x, int(para['e']), int(para['n']))
        # y_bin = format(y, 'b')
        y_hex = hex(y)[2:]
        # The r=seed-length most significant bit is used for the next iteration
        x = int(y_hex[:seedLen_hex], 16)
        # The k least significant bits is used for the hex sequence
        hexSeq += y_hex[-k_hexlength:]

        # Print progress...
        if i%100 == 0:
            print(f"Generating {i} of {m}\nseed: {para['seed']}\nhexSeq: {hexSeq}")

        # If hexSeq is equal to or more than required key-length, output keys
        if len(hexSeq) >= keyLen_hex:
            end = 0
            for j in range(0, len(hexSeq) - keyLen_hex, keyLen_hex):
                randomNumList.append(int(hexSeq[j: j + keyLen_hex], 16))
                # If N random numbers are generated, return list of numbers
                if len(randomNumList) == N:
                    return randomNumList
                end = j
            # Assign leftover bits to the bit sequence
            hexSeq = hexSeq[end:]


def prngControl(N, KEY_BITLENGTH, para):
    """
    :param N: number of random number to generate
    :param KEY_BITLENGTH: padded bit-length of outputs
    :param para: series containing parameters seed, n, phi, p, q, e, d, k
    :return:
    """
    randomNumList = []
    random.seed(para['seed'])
    for _ in range(N):
        randomNumList.append(random.getrandbits(KEY_BITLENGTH))
    return randomNumList


def split_equalSize(data, n):
    chunkSize = int(math.ceil(len(data)/n))
    for i in range(0, len(data), chunkSize):
        yield data[i: i + chunkSize]


def generatePara(SEED_BITLENGTH, PRIME_BITLENGTH):
    # ----- Generate p, q, phi, n
    while True:
        p = sympy.randprime(0, pow(2, PRIME_BITLENGTH) - 1)
        if p%4 == 3:
            break
    while True:
        q = sympy.randprime(0, pow(2, PRIME_BITLENGTH) - 1)
        if q%4 == 3:
            break
    n = p*q
    phi = (p - 1)*(q - 1)

    # ----- Generate public & private keys
    while True:
        e = random.randint(2, phi)
        if math.gcd(e, phi) == 1:
            break
    d = sympy.mod_inverse(e, phi)

    # ----- Generate k for RSA PRNG
    k = PRIME_BITLENGTH*2 - SEED_BITLENGTH
    assert SEED_BITLENGTH*e >= 2*PRIME_BITLENGTH*2

    # ----- Generate seed
    while True:
        seed = sympy.randprime(0, pow(2, SEED_BITLENGTH) - 1)
        if math.gcd(seed, n) == 1:
            break
    # parameters = pd.Series([seed, n, phi, p, q, e, d, k], index='seed n phi p q e d k'.split()).astype(int)
    parameters = {
        'seedLen': SEED_BITLENGTH,
        'seed'   : seed,
        'n'      : n, 'phi': phi, 'p': p, 'q': q, 'e': e, 'd': d, 'k': k
    }
    return parameters


# %% Perform Entire Experiement
PRIME_BITLENGTH = 1024
seedLenList = [256, 512, 1024]
keyLenList = [16, 32, 64]
prngList = [prngBBS, prngHash, prngRSA, prngControl]
N = 1_000_000

# ----- Generate Parameters for Experiement
paraExpList = []
for prng in prngList:
    for keyLen in keyLenList:
        for seedLen in seedLenList:
            para = generatePara(seedLen, PRIME_BITLENGTH)
            para['keyLen'] = keyLen
            para['prng'] = prng
            paraExpList.append(para)


# ----- Perform Experiment (Multi-processing)
def exp(N, para):
    prng = para['prng']
    seedLen = para['seedLen']
    keyLen = para['keyLen']

    print(
        f"\n===== {prng.__name__} | seed-length {seedLen}bits | key-length {keyLen}bits =====")
    startTm_exp = dt.now()
    randomNumList = prng(N, keyLen, para)
    processTm = (dt.now() - startTm_exp).total_seconds()/N

    # Compute max key count
    keyCount = pd.Series(randomNumList).value_counts()
    maxCount_perc = keyCount.max()/keyCount.sum()

    return [prng.__name__, keyLen, seedLen, maxCount_perc, processTm]


startTm = dt.now()
nExe = len(paraExpList)
with concurrent.futures.ProcessPoolExecutor() as exe:
    resultGen = exe.map(exp, [N]*nExe, paraExpList)

    resultList = []
    for res in resultGen:
        resultList.append(res)

results = pd.DataFrame(resultList,
                       columns='PRNG keyLength seedLength maxCount_perc processTime'.split())
results.to_csv(f"data/results.csv", index=False)

dur = dt.now() - startTm
print(f"Completed in {str(dur)}seconds ")
os.system('say "Complete"')

# %% Plot correlation heatmap
FIG_SIZE = 5
plotConfig = {'nrows': 2, 'ncols': 2}
fig = plt.figure(figsize=(FIG_SIZE*1.5*plotConfig['ncols'], FIG_SIZE*plotConfig['nrows']),
                 dpi=300)

results = pd.read_csv(f"data/results.csv")

for k, prng in enumerate(results['PRNG'].unique()):
    ax = plt.subplot(plotConfig['nrows'], plotConfig['ncols'], k + 1)
    onePRNG = results[results['PRNG'] == prng]
    sns.heatmap(onePRNG.corr(), annot=True, center=0, cmap="PiYG", ax=ax)
    ax.set_title(f"{prng}")
plt.show()
fig.savefig(f"data/results_heatmap.png")

# %% Line plots
sns.set_style('darkgrid')
sns.set_context('notebook')
FIG_SIZE = 5
plotConfig = {'nrows': 2, 'ncols': 2}
fig = plt.figure(figsize=(FIG_SIZE*1.5*plotConfig['ncols'], FIG_SIZE*plotConfig['nrows']),
                 dpi=300)

style = '- - - --'.split()
linewidth = 3
marker = 'o'
markersize = linewidth*2

for k, keyLen in enumerate(keyLenList):
    oneKeylen = results[results['keyLength'] == keyLen]

    results_byPrng = pd.DataFrame(index=results['seedLength'].unique())
    for prng in prngList:
        onePRNG = oneKeylen[oneKeylen['PRNG'] == prng.__name__].set_index('seedLength')
        results_byPrng[prng.__name__] = onePRNG['maxCount_perc']

    ax = plt.subplot(plotConfig['nrows'], plotConfig['ncols'], k + 1)
    results_byPrng.plot.line(
        marker=marker, markersize=markersize,
        linewidth=linewidth, style=style,
        ax=ax
    )
    ax.set_xlabel('Seed Length')
    ax.set_ylabel('Percentage of Maximum Repeated Key')
    ax.set_xlim(200, 1100)
    ax.set_title(f"{keyLen}bit Key-length")

plt.savefig(f"data/results_maxCount.png")
plt.show()

# %% Examine Security Stability of PRNG (Multi-processing) with varying seed-length
PRIME_BITLENGTH = 1024
seedCount = os.cpu_count()
SEEDLENGTH_LIST = [256, 512, 1024]
KEYLENGTH_LIST = [16, 32]
KEY_BITLENGTH = 16
N = 1_000_000
prng = prngHash

# ----- Generate parameters
paraList = []
for prng in prngList:
    for keyLen in keyLenList:
        for seedLen in SEEDLENGTH_LIST:
            paraSubList = []
            while len(paraSubList) < seedCount:
                para = generatePara(seedLen, PRIME_BITLENGTH)
                # Prevent the repetition of seeds
                if para['seed'] in [pr['seed'] for pr in paraSubList]:
                    continue
                else:
                    para['keyLen'] = keyLen
                    para['prng'] = prng
                    paraSubList.append(para)
            paraList += paraSubList

# ----- Generate keys
startTm = dt.now()
maxCount = pd.DataFrame()
# for key_bitlength in KEYLENGTH_LIST:
#     for seedLen in SEEDLENGTH_LIST:

# ----- Generate keys
print(f"\n===== Seed-length {seedLen} =====")
nExe = len(paraList)
with concurrent.futures.ProcessPoolExecutor() as executor:
    resultGen = exe.map(exp, [N]*nExe, paraList)

    resultList = []
    for res in resultGen:
        resultList.append(res)

results = pd.DataFrame(resultList,
                       columns='PRNG keyLength seedLength maxCount_perc processTm'.split())
results.to_csv(f"data/results_{seedCount}each.csv", index=False)

# %% ----- Plot stability results
sns.set_style('darkgrid')
sns.set_context('notebook')
FIG_SIZE = 5
NCOLS = 2
plotConfig = {'nrows': math.ceil(len(maxCount.columns)/NCOLS), 'ncols': NCOLS}
fig = plt.figure(figsize=(FIG_SIZE*1.5*plotConfig['ncols'], FIG_SIZE*plotConfig['nrows']),
                 dpi=300)
errwidth = 0

for k, ((seedLen, maxCnts), colour) in enumerate(
        zip(maxCount.iteritems(), PRNG_COLOURS[prng.__name__])):
    plt.subplot(plotConfig['nrows'], plotConfig['ncols'], k + 1)
    ax = sns.barplot(range(seedCount), maxCnts, color=colour, errwidth=0)
    ax.set_title(
        f"Percentage of Highest Key-count | {prng.__name__}\n{seedLen}-bit Seeds | {KEY_BITLENGTH}-bit Keys")
    ax.set_xlabel(f"Seed")
    ax.set_xticklabels([])
    ax.set_ylabel("Percentage")

    plt.tight_layout()
    plt.savefig(f"data/{prng.__name__}_maxCount_keyLen{KEY_BITLENGTH}.png")
    plt.show()

dur = dt.now() - startTm
print(f"Completed in {str(dur)}seconds ")
os.system('say "Complete"')
