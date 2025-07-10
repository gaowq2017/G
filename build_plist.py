import plistlib

# Create a dictionary with your data
data = {
    'som_positive':
        {
            'defectName': ['bianyuanluosi', 'zhutiluosi', 'daoxiangcao'],
            'bianyuanluosi': [[260, 430, 200, 200], [1050, 260, 200, 200], [1500, 240, 200, 200], [1070, 260, 200, 200],
                              [2400, 500, 200, 200], [250, 2100, 200, 200], [2360, 2100, 200, 200], [2040, 2650, 200, 200]],
            'zhutiluosi': [[700, 930, 300, 300], [1870, 930, 300, 300], [700, 1600, 300, 300], [1870, 1600, 300, 300],
                           [980, 1040, 300, 300], [1570, 1040, 300, 300], [980, 1500, 300, 300], [1580, 1500, 300, 300],
                           [1080, 2370, 300, 300], [1450, 2370, 300, 300]],
            'daoxiangcao': [[1970, 2460, 500, 256]]
        },
    'som_negative':
        {
            'defectName': ['daoxiangcao'],
            'daoxiangcao': [[1300, 1890, 400, 200]]
        },
    'pdb_positive':
        {
            'defectName': ['dxcleft', 'dxcright'],
            'dxcleft': [[50, 520, 100, 300], [50, 1020, 100, 300]],
            'dxcright': [[420, 520, 100, 300], [420, 1020, 100, 300]],
        },
    'pdb_negative':
        {
            'defectName': ['maozi'],
            'maozi': [[30, 910, 200, 200], [510, 910, 200, 200]],
        },
    'nic_positive':
        {
            'defectName': ['daoxiangcao1', 'daoxiangcao2', 'daoxiangcao3', 'zhuzi'],
            'daoxiangcao1': [[800, 640, 200, 200], [1480, 680, 200, 200], [2140, 660, 200, 200], [2750, 650, 200, 200],
                             [3360, 670, 200, 200], [4030, 670, 200, 200]],
            'daoxiangcao2': [[4000, 900, 200, 200]],
            'daoxiangcao3': [[1800, 1250, 200, 200]],
            'zhuzi': [[50, 530, 400, 400], [50, 1400, 400, 400]],
        },
    'nic_negative':
        {
            'defectName': ['heimao', 'luosi', 'tiepian'],
            'heimao': [[1700, 180, 200, 200], [2560, 180, 200, 200], [470, 350, 200, 200], [3460, 520, 200, 200],
                       [3850, 430, 200, 200], [4150, 390, 200, 200], [800, 720, 200, 200], [1130, 910, 200, 200],
                       [2760, 910, 200, 200], [4150, 1090, 200, 200], [850, 1310, 200, 200], [1470, 1390, 200, 200],
                       [2100, 1310, 200, 200], [2760, 1390, 200, 200], [3400, 1310, 200, 200], [620, 1750, 200, 200],
                       [1280, 1750, 200, 200], [2970, 1750, 200, 200], [3610, 1740, 200, 200], [330, 1900, 200, 200],
                       [1680, 1900, 200, 200], [2570, 1900, 200, 200], [3910, 1900, 200, 200], [100, 380, 200, 200],
                       [100, 1100, 200, 200]],
            'luosi': [[60, 230, 300, 200], [60, 1240, 300, 200]],
            'tiepian': [[60, 100, 300, 1500]],
        },
    'bmc_negative':
        {
            'defectName': ['luosi', 'tiepian'],
            'luosi': [[270, 330, 100, 100], [520, 330, 100, 100], [270, 750, 100, 100], [510, 750, 100, 100]],
            'tiepian': [[230, 340, 400, 500]]
        },
    'display_positive':
        {
            'defectName': ['daoxiangcao'],
            'daoxiangcao': [[330, 210, 360, 300]],
        },
}

# Specify the file path where you want to save the .plist file
plist_file_path = 'config.plist'

# Write the dictionary to the .plist file
with open(plist_file_path, 'wb') as plist_file:
    plistlib.dump(data, plist_file)

print(f'Plist data saved to {plist_file_path}')
