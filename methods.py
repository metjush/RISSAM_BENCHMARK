import numpy as np
import pandas as pd
import requests as req
from defaults import *
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objs as go

def konsoliduj_BA(prijmy=None, vydavky=None):
    """
    Metoda na konsolidaciu Bratislavy - magistrat + mestske casti 

    """
    if prijmy is not None:
        filter_prijmy = prijmy[prijmy.id_samosprava.isin(BA_SAMOSPRAVY)]
        preneseny_vykon = filter_prijmy[(filter_prijmy.kodKlasifikacie == '312012') & (filter_prijmy.organizacia == 'Hlavne mesto Slovenskej republiky Bratislava')]
        filter_prijmy_net = filter_prijmy.drop(preneseny_vykon.index.values[0], axis=0)
        filter_prijmy_net.id_samosprava = BA_KONS
        filter_prijmy_net.samosprava = 'Bratislava - konsolidovana'
        filter_prijmy_net.loc[(filter_prijmy_net.organizacia.str.contains('Mestská časť')) | (filter_prijmy_net.organizacia == 'Hlavne mesto Slovenskej republiky Bratislava'), 'organizacia'] = 'Bratislava - urady'
        prijmy = prijmy.append(filter_prijmy_net, ignore_index=True)

    if vydavky is not None:
        filter_vydavky = vydavky[vydavky.id_samosprava.isin(BA_SAMOSPRAVY)]
        preneseny_vykon = filter_vydavky[(filter_vydavky.kodKlasifikacie == '641013') & (filter_vydavky.organizacia == 'Hlavne mesto Slovenskej republiky Bratislava')]
        filter_vydavky_net = filter_vydavky.drop(preneseny_vykon.index.values[0], axis=0)
        filter_vydavky_net.id_samosprava = BA_KONS
        filter_vydavky_net.samosprava = 'Bratislava - konsolidovana'
        filter_vydavky_net.loc[(filter_vydavky_net.organizacia.str.contains('Mestská časť')) | (filter_vydavky_net.organizacia == 'Hlavne mesto Slovenskej republiky Bratislava'), 'organizacia'] = 'Bratislava - urady'
        vydavky = vydavky.append(filter_vydavky_net, ignore_index=True)

    return prijmy, vydavky


def konsoliduj_KE(prijmy = None, vydavky=None):
    """
    Metoda na konsolidaciu Kosic - magistrat + mestske casti 

    """
    if prijmy is not None:
        filter_prijmy = prijmy[prijmy.id_samosprava.isin(KE_SAMOSPRAVY)]
        filter_prijmy.id_samosprava = KE_KONS
        filter_prijmy.samosprava = 'Košice - konsolidovane'
        filter_prijmy.loc[(filter_prijmy.organizacia.str.contains('Mestská časť')) | (filter_prijmy.organizacia == 'Mesto Košice'), 'organizacia'] = 'Košice - urady'
        prijmy = prijmy.append(filter_prijmy, ignore_index=True)

    if vydavky is not None:
        filter_vydavky = vydavky[vydavky.id_samosprava.isin(KE_SAMOSPRAVY)]
        transfery = filter_vydavky[(filter_vydavky.kodKlasifikacie.isin(['641013','641009'])) & (filter_vydavky.organizacia == 'Mesto Košice')]
        filter_vydavky_net = filter_vydavky.drop(transfery.index.values, axis=0)
        filter_vydavky_net.id_samosprava = KE_KONS
        filter_vydavky_net.samosprava = 'Košice - konsolidovane'
        filter_vydavky_net.loc[(filter_vydavky_net.organizacia.str.contains('Mestská časť')) | (filter_vydavky_net.organizacia == 'Mesto Košice'), 'organizacia'] = 'Košice - urady'
        vydavky = vydavky.append(filter_vydavky_net, ignore_index=True)

    return prijmy, vydavky


def obce(base):
    """
    Metoda ktora vola RIS SAM API a stahuje zoznam samosprav 
    """
    call = base + "ciselnik/organizacie/" 
    response = req.get(call)
    okresy = pd.DataFrame(response.json()['payload'])

    samospravy = None
    for okres in okresy.kodOrganizacie:
        call = base + "ciselnik/organizacie/" + okres
        response = req.get(call)
        if samospravy is None:
            samospravy = pd.DataFrame(response.json()['payload'])
        else:
            samospravy = samospravy.append(pd.DataFrame(response.json()['payload']), ignore_index=True)
    
    return samospravy

def org(samospravy, base):
    """
    Metoda ktora vola RIS SAM API a stahuje zoznam vsetkych samospravnych organizacii
    """
    call = base + "ciselnik/organizacie/"
    organizacie = None
    
    for okres in pd.unique(samospravy.kodNadurovne):
        for obec in samospravy[samospravy.kodNadurovne == okres].kodOrganizacie:
            call_obec = call + obec
            response = req.get(call_obec)
            if organizacie is None:
                organizacie = pd.DataFrame(response.json()['payload'])
            else:
                organizacie = organizacie.append(pd.DataFrame(response.json()['payload']), ignore_index=True)
        print(okres)
        organizacie.to_pickle('organizacie_'+okres+'.pkl')

    return organizacie 

def budget_call(base, org, PV, klasifikacia, rok):
    # pomocna funkcia pre volanie rozpoctu konkretnej organizacie
    return base + 'rozpocet/' + org + '/prehlad-klasifikacii/' + PV + '/' + klasifikacia + '/' + str(rok)

def dlh_call(base, org, rok):
     # pomocna funkcia pre volanie dlhu konkretnej organizacie
    return base + 'rozpocet/' + org + '/dlhy-zavazky/' + str(rok)

def budget_builder(base, org, PV, klasifikacia, roky=np.arange(2015,2021)):
    """
    Metoda na zostavenie komplet rozpoctu organizacii za uvedene roky
    org je iba zoznam IDciek a zoznam nadurovni aj s menom ako np array
   
    """
    master = None
    for rok in roky:
        for polozka in org:
            try:
                r = req.get(budget_call(base, polozka[0], PV, klasifikacia, rok))
                d = pd.DataFrame(r.json()['payload'])
            except:
                continue
            d['kodNaduroven'] = polozka[1]
            d['samosprava'] = polozka[2]
            if master is None:
                master = d
            else:
                master = master.append(d, ignore_index=True)
        master.to_pickle(f"BUDGET_{PV}.pkl")

    return master


def master_budget_builder(base, klasifikacie, MC, roky=np.arange(2015,2021)):
    """
    Wrapper metoda pre zostavenie rozpoctu viacerych klasifikacii 
    """

    pickles = []
    for pv in klasifikacie.keys():
        for klasifikacia in klasifikacie[pv]:
            nazov = 'rozpocet_' + pv + '_' + klasifikacia + '.pkl'
            data = budget_builder(base, MC, pv, klasifikacia, roky)
            data.to_pickle(nazov)
            print('Rozpocet ' + nazov + ' uspesne ulozeny ako pickle')
            pickles.append(nazov)
    return pickles

def budget_editor(pickles):
    for p in pickles:
        df = pd.read_pickle(p)
        df['skutocnost_per_capita'] = df['skutocnost'] / df['Pocet_obyvatelov']
        df['dlh_per_capita'] = df.Dlh_eur / df['Pocet_obyvatelov']
        df['log_capita'] = np.log10(df.Pocet_obyvatelov.values)
        df.to_pickle(p)
        print(p + "uspesne upravene a ulozene")
    return pickles

def pivotka(name, filetype='pkl', value='skutocnost', mode='sum'):
    """
    Metoda na vytvorenie pivot tabulky z podkladovej rozpoctovej tabulky pre dalsiu analyzu 
    """
    if filetype == 'pkl':
        df = pd.read_pickle(name)
    else:
        df = pd.read_csv(name)
    df[value] = df[value].astype(np.float64)
    rel = df.groupby(['rok','kodNaduroven','samosprava','kodOrganizacie','nazovOrganizacie','kodKlasifikacie'])[value]
    if mode == 'sum':
        rel = rel.sum().reset_index()
    elif mode == 'mean':
        rel = rel.mean().reset_index()
    else:
        rel = rel.sum().reset_index()
        print('Neznamy mod, pouzivam sucet')

    rel.columns = ['rok','id_samosprava','samosprava','id_organizacia','organizacia','kodKlasifikacie',value]

    rel_pivot = rel.pivot(index=['id_samosprava','samosprava','id_organizacia','organizacia','kodKlasifikacie'], columns='rok', values=value).reset_index().fillna(0.)
    return rel_pivot


def pivot_filter(pivot, filter_plus, filter_minus=None, mode='sum', level='samosprava', skip_MC=None, net_BA=False, net_BA_values=None):
    """
    Metoda na vyber konkretnych poloziek z pivot tabulky 
    Filter_plus = ktore polozky zahrnut do filtra
    Filter_minus = ktore polozky odpocitat od sumarizacie
    net_BA, net_BA_values = ake sumy odratat od BA magistratu (Transfery MC)
    """
    if skip_MC is not None:
        pivot = pivot[~pivot[level].isin(skip_MC)]

    baname = (int(BA_UROVEN),'Bratislava')
    levels = [f"id_{level}",level]
    if level != 'samosprava':
        baname = (int(HMBA_URAD),'Hlavne mesto Slovenskej republiky Bratislava')
        pivot = pivot[(pivot[level] == 'Bratislava - urady') | (pivot['samosprava'] != 'Bratislava - konsolidovana')]
        pivot = pivot[(pivot[level] == 'Košice - urady') | (pivot['samosprava'] != 'Košice - konsolidovane')]

    pluska = pivot[pivot.kodKlasifikacie.isin(filter_plus)].groupby(levels)
    if mode == 'sum':
        pluska = pluska.sum()
    elif mode == 'mean':
        pluska = pluska.mean()
    else:
        pluska = pluska.sum()
        print('Neznamy mod, pouzivam sucet')

    pluska.columns = ['id','2015','2016','2017','2018','2019','2020','2021']
    pluska.drop(['id'],axis=1,inplace=True)

    if net_BA:
         pluska.loc[pluska.index == baname] = pluska.loc[pluska.index == baname] - net_BA_values

    if filter_minus is not None:
        minuska = pivot[pivot.kodKlasifikacie.isin(filter_minus)].groupby(levels)
        if mode == 'sum':
            minuska = minuska.sum()
        elif mode == 'mean':
            minuska = minuska.mean()
        else:
            minuska = minuska.sum()
            print('Neznamy mod, pouzivam sucet')
        minuska.columns = ['id','2015','2016','2017','2018','2019','2020','2021']
        minuska.drop(['id'],axis=1,inplace=True)

        rozdiel = pluska.merge(minuska, left_index=True, how='left', right_index=True).fillna(0)
        cols = pluska.columns 
        for col in cols:
            rozdiel[col] = rozdiel[col+'_x'] - rozdiel[col+'_y']
            rozdiel.drop([col+'_x', col+'_y'], axis=1, inplace=True)

    else:
        rozdiel = pluska.loc[:,'2015':'2021']
    
    return rozdiel

def nice_line_chart(df, Y, title_name, ynames, xtit='', ytit="",xform=',', yform=',', ylim=None):
    """
    Metoda na vytvorenie ciaroveho grafu
    """
    fig = go.Figure()

    for i, var in enumerate(ynames):
        try:
            col = colordict[var]
        except:
            col = 'skyblue'

        fig.add_trace(
            go.Scatter(
                name=var,
                hovertext=var,
                mode='lines',
                y=df[Y[i]],
                x=df.index,
                line=dict(
                    color=col
                )
                
            )
        )

    fig.update_layout(
        yaxis_title=ytit,
        xaxis_title=xtit,
        title=title_name,
        margin=dict(
            l=10,
            r=10,
            b=15,
            t=25),
        height=600,
        plot_bgcolor='rgba(230,230,230,0)',
        font_family='Inter',
        font_size=10
    )

    fig.update_yaxes(
        range=ylim,
        tickformat=yform,
        showgrid=True,
        gridcolor='gainsboro'
    )

    fig.update_xaxes(
        tickformat=xform,
        showgrid=True,
        gridcolor='gainsboro'
    )

    fig.show()

def colorizer(row):
    """
    Pomocna metoda na zafarbenie bodov na scatter plote
    """
    mask = np.array(row) != 'skyblue'
    order = np.arange(1,len(row)+1,1)
    index = np.max(mask * order)
    return row[index - 1]

def nice_chart(df, x, y, title_name, hover_name, xtit="", ytit="",xform=',', yform=',', xlim=None, ylim=None):
    """
    Metoda na vytvorenie scatter plotu
    """
    df['Bratislava'] = df[hover_name].str.contains('Bratislava').apply(lambda x: 'orangered' if x else 'skyblue')
    df['Kosice'] = df[hover_name].str.contains('Košice').apply(lambda x: 'goldenrod' if x else 'skyblue')
    df['Krajske'] = df[hover_name].isin(krajske_mesta).apply(lambda x: 'YellowGreen' if x else 'skyblue')
    df['Color'] = df[['Krajske','Bratislava','Kosice']].apply(colorizer, axis=1)

    df['Size'] = df[hover_name].isin(krajske_mesta).apply(lambda x: 8 if x else 4)

    fig = go.Figure([
        go.Scatter(
            name=y,
            x=df[x],
            y=df[y],
            mode='markers',
            marker=dict(
                color=df['Color'],
                size=df['Size']),
            hovertext=df[hover_name]
        )
    ])

    fig.update_layout(
        yaxis_title=ytit,
        xaxis_title=xtit,
        title=title_name,
        margin=dict(
            l=10,
            r=10,
            b=15,
            t=25),
        height=600,
        plot_bgcolor='rgba(230,230,230,0)',
        font_family='Inter',
        font_size=10
    )

    fig.update_yaxes(
        range=ylim,
        tickformat=yform,
        showgrid=True,
        gridcolor='gainsboro'
    )

    fig.update_xaxes(
        range=xlim,
        tickformat=xform,
        showgrid=True,
        gridcolor='gainsboro'
    )

    fig.show()


def benchmark_bake(df, isolate=False, kons=False):
    """
    Priprava na benchmarking, vytvorenie filtrovacej mapy pre vyber konkretnych samosprav
    Bud vsetky konsolidovane (BA a KE) alebo vsetky ostatne
    Bud izolacia (iba tie) alebo vylucenie (vsetky okrem) 
    """
    ix = df.index.droplevel('id_samosprava')
    if kons:
        try:
            bake_map = ix.str.contains('konsolid').values
        except AttributeError:
            bake_map = ix.str.contains('konsolid')
    else:
        try:
            bake_map = ix.str.contains('Bratislava').values
            bake_map = bake_map + ix.str.contains('Košice').values
        except AttributeError:
            bake_map = ix.str.contains('Bratislava')
            bake_map = bake_map + ix.str.contains('Košice')
    bake_map = bake_map.astype(np.bool8)
    
    if isolate:
        return bake_map
    return np.invert(bake_map)

def benchmarking(target, capita):
    """
    Metoda na vytvorenie benchmarkingu pre vybrany rozpocet
    """
    df = target.merge(capita, how='left', on='id_samosprava')
    df.index = target.index
    df['log_rounded'] = np.floor(df.log_obyv).fillna(0.).astype(int)
    df_bez_BAKE = df[benchmark_bake(df)]
    df_benchmark = df_bez_BAKE.groupby(['log_rounded']).median().loc[:,'2015':'2021']
    df_BAKE_kons = df.loc[benchmark_bake(df, True, True), '2015':'2021']
    df_BAKE_kons = df_BAKE_kons.droplevel('id_samosprava', axis=0)
    df_benchmarking = pd.concat([df_benchmark, df_BAKE_kons], axis=0)
    return df_benchmarking.T


### METODY PRE FINSTAT, ZATIAL NEPOUZIVANE

def finstat_vykaz(ICO):
    URL = FINSTAT_VZS(ICO)
    scrape = req.get(URL)
    soup = BeautifulSoup(scrape.content, 'html.parser')
    title = soup.find_all('title')[0].text
    title = title.replace(' - Výkaz ziskov a strát','')
    
    tabulka = soup.find_all("table", class_="data-table-main")[0]
    converted = pd.read_html(tabulka.prettify(), header=0)
    df = converted[0]
    df.drop([0,1,2], inplace=True)
    df = df.loc[:,:'2015']
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c].str.replace('€','').str.replace('\s+',''))
    df = df.rename({df.columns[0]: 'Polozka'}, axis=1)
    df['ICO'] = ICO
    df['MC'] = title
    return df

def finstat_suvaha(ICO):
    URL = FINSTAT_SUVAHA(ICO)
    scrape = req.get(URL)
    soup = BeautifulSoup(scrape.content, 'html.parser')
    title = soup.find_all('title')[0].text
    title = title.replace(' - Súvaha','')
    
    tabulka = soup.find_all("table", class_="data-table-main")[0]
    converted = pd.read_html(tabulka.prettify(), header=0)
    df = converted[0]
    df.drop([0,1], inplace=True)
    df = df.loc[:,:'2015']
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c].str.replace('€','').str.replace('\s+',''))
    df = df.rename({df.columns[0]: 'Polozka'}, axis=1)
    df['ICO'] = ICO
    df['MC'] = title
    return df

def finstat_master(ICOlist, vykaz=True):
    master = None
    finstat = lambda x, i: finstat_vykaz(i) if x else finstat_suvaha(i)
    for ico in ICOlist:
        print(ico)
        slave = finstat(vykaz, ico)
        if master is None:
            master = slave
        else:
            master = master.append(slave, ignore_index=True)
    pklname = 'finstat_' + ['suvaha','vykaz'][int(vykaz)] + '_MC.pkl'
    master.to_pickle(pklname)
    return master
    
