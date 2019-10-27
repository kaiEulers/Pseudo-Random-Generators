"""
@author: kaisoon
"""
# ----- Colour Palettes
import seaborn as sns

# Seaborn default palette
sns = dict(zip(
    'blue orange green red purple brown pink gray yellow teal'.split(),
    sns.color_palette().as_hex()
))

# ColourBrewer Qualitative Paired
cbPaired = dict(zip(
    'lightBlue blue lightGreen green lightRed red lightOrange orange lightPurple purple yellow brown'.split(),
    ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
     '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
))

# ColourBrewer Qualitative Dark2
cbDark2 = dict(zip(
    'blueGreen orange purple pink green yellow brown gray'.split(),
    ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
))

# ColourBrewer Qualitative Set1
cbSet1 = dict(zip(
    'red blue green purple orange yellow brown pink gray'.split(),
    ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628',
     '#f781bf', '#999999']
))

# ColourBrewer Qualitative Set2
cbSet2 = dict(zip(
    'blueGreen orange blue pink green yellow beige gray'.split(),
    ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
     '#e5c494', '#b3b3b3']
))

# ColourBrewer Qualitative Pastel
cbPas1 = dict(zip(
    'red blue green purple orange yellow brown pink'.split(),
    ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd',
     '#fddaec']
))

# ColourBrewer Sequential Palette
# Yellow to Green
cbYlGn = ['#ffffcc', '#c2e699', '#78c679', '#238443']

# ColourBrewer Sequential Palette
# Grays
cbGrays = ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252',
           '#252525', '#000000']
