API_BASE = "https://rozpocet.sk/api/sam/"
BA_UROVEN = "582000"
BA_KONS = 555555555
KE_UROVEN = "599981"
KE_KONS = 999999999
HMBA_URAD = "00603481"

krajske_mesta = [
    'Bratislava',
    'Košice',
    'Bratislava - konsolidovana',
    'Košice - konsolidovane',
    'Trenčín',
    'Trnava',
    'Žilina',
    'Nitra',
    'Banská Bystrica',
    'Prešov'
]

colordict = {
    'Bratislava': 'orangered',
    'Košice': 'goldenrod',
     '<100 ob.':'lightskyblue',
     '100 - 1 000 ob.':'skyblue', 
     '1 000 - 10 000 ob.':'deepskyblue', 
     '10 000+ ob.': 'cornflowerblue',
     'Byty a NP': 'deepskyblue',
     'Stavby': 'cornflowerblue',
     'Pozemky': 'skyblue'
}

BA_SAMOSPRAVY = [
    582000,
    528595,
    529311,
    529320,
    529338,
    529346,
    529354,
    529362,
    529371,
    529389,
    529397,
    529401,
    529419,
    529427,
    529435,
    529443,
    529460,
    529494
    ]

KE_SAMOSPRAVY = [
    599981,
    598119,
    598127,
    598151,
    598186,
    599875,
    599891,
    598194,
    598208,
    598216,
    598224,
    599841,
    599859,
    599883,
    599972,
    598682,
    599018,
    599093,
    599786,
    599794,
    599816,
    599824,
    599913
]

KLASIFIKACIE = {
    'prijmy': ['ekp','zdr'],
    'vydavky': ['ekv','fnc','zdr']
}

FINSTAT_BASE = "https://finstat.sk/"
FINSTAT_SUVAHA = lambda x: FINSTAT_BASE + x + "/suvaha"
FINSTAT_VZS = lambda x: FINSTAT_BASE + x + "/vykaz_ziskov_strat"

PICKLES = ['rozpocet_prijmy_ekp.pkl','rozpocet_prijmy_zdr.pkl','rozpocet_vydavky_ekv.pkl','rozpocet_vydavky_fnc.pkl','rozpocet_vydavky_zdr.pkl']
HR = ['610','620','637027','642012','642013']
BEZNE = ['100','210','220','240','250','290','310','331']
KAPITALOVE = ['230','320','332']
PREVADZKA = [
    '610',
    '620',
    '632',
    '633',
    '637',
    '641',
    '642004',
    '642005',
    '642012',
    '642013',
    '650'
]

PREVADZKA_BEZ_SKOL = [
    '610',
    '620',
    '632',
    '633',
    '637',
    '641',
    '642012',
    '642013',
    '650'
]

RO = ['00490873','00641405','00896276','30775205','30779278','30791898',
'30842344','31755534','31768857','31769403','31780334','31780725',
'31780873','31810209','31810519','31811027','31816088','31816118',
'36067211','36067253','36070939','36071323','36071331','37926012','42174970']