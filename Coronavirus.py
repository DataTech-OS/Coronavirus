from selenium import webdriver
from selenium.webdriver.remote.errorhandler import NoSuchElementException, ElementNotInteractableException, ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options
from sklearn.linear_model import LinearRegression
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from bs4 import BeautifulSoup as bs
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import json
import csv
import os
import re

pd.options.mode.chained_assignment = None

N = 60000000
cnts = ["Austria", "Lettonia", "Belgio",	"Lituania", "Bulgaria",	
       "Lussemburgo", "Repubblica Ceca", "Malta", "Cipro", "Paesi Bassi", 
       "Croazia", "Polonia", "Danimarca", "Portogallo", "Estonia", 
       "Romania", "Finlandia", "Slovacchia", "Francia", "Slovenia", "Germania", 
       "Spagna", "Grecia", "Svezia", "Irlanda", "Ungheria", "Italia"]
regions = ["Valle d'Aosta", "Piemonte", "Liguria", "Lombardia", "Trentino Alto Adige", "Veneto", "Friuli Venezia Giulia", "Emilia Romagna", "Toscana", "Umbria", "Marche", "Lazio", "Abruzzo", "Molise", "Campania", "Puglia", "Basilicata", "Calabria", "Sicilia", "Sardegna"]
url = "https://lab24.ilsole24ore.com/coronavirus/"
path = "C:/Users/111ol/chromedriver_win32/chromedriver.exe"
root = "C:/Users/111ol/Desktop/Test/"

def SIR(t, y, beta, gamma):
    	S = y[0]
    	I = y[1]
    	return([-beta*S*I/60000, beta*S*I/60000-gamma*I, gamma*I])

def sumsq(p, initial_state, s, sus, inf):
    beta, gamma = p
    sol = solve_ivp(SIR,[0,s],initial_state,t_eval=np.arange(0,s,1), args=(beta,gamma))    
    l1 = sum((sol.y[0] - sus)**2)
    l2 = sum((sol.y[1] - inf)**2)
    return l1 + l2

def rename_file(region):
    file = "C:/Users/111ol/Desktop/Test/Italia-regioni-province.json"
    r, fil = os.path.split(file)
    newfile = r + "/" + region + fil
    os.rename(file, newfile)

def average(series, l):
    s = series.size
    div = l
    if s < l:
        div = float(s)
    res = sum(series[s - int(div):].astype(float)) / div
    return res

def moving_average(series, l=7.0):
    return [average(series[:i+1],l) for i in range(series.size)]   
    
#set options for the Chrome browser:
#start in the background (headless)
#set default download directory

chrome_options = Options()
prefs = {"profile.default_content_settings.popups": 0, "download.default_directory": r"C:\Users\111ol\Desktop\Test\\", "directory_upgrade": True} 
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--headless")
#chrome_options.add_argument("--incognito")
#chrome_options.add_argument("--window-size=960x540")

#start the browser
browser = webdriver.Chrome(options=chrome_options, executable_path=path)
browser.implicitly_wait(5)

#allow for downloads in headless mode, otherwise we'll get a download error
browser.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': r"C:\Users\111ol\Desktop\Test\\"}}
command_result = browser.execute("send_command", params)

#retrieve the html code of the starting web page and find all the iframes
browser.get(url)
bsobj = bs(browser.page_source, 'lxml')
table = bsobj.findAll("iframe", {"src" : True})
table.extend(bsobj.findAll("iframe", {"data-src":True}))
sources = []

#we just want html source code and not php or other things
for element in table:
    try:
        e = element.attrs["src"]
    except KeyError:
        e = element.attrs["data-src"]
    if not e.endswith("php") and not e.endswith("embed"):
        if e.startswith("//"):
            e = "https:" + e
        sources.append(e)  
    
#now we retrieve the data in each page
tables = []
i = -1
for page in sources:
    #data is kind of badly formatted in this page and overwrites
    #a previous downloaded file with the correctly formatted data
    if "italia-varizione-giornaliera" in page or "lombardia-chiamate" in page:
        continue
    #this is instead a duplicate
    if "settimanale" in page:
        continue
    
    browser.get(page)
   
    #the website lets us download data by clicking on a button 
    try:    
        download = browser.find_element_by_id("scarica-dati")
    except NoSuchElementException:
        download = None
    
    #not every page has this option, some pages are html tables
    if download:        
        try:
            reg = browser.find_element_by_class_name("ss-single-selected")
        except NoSuchElementException:
            reg = None

        if reg:
            opt_idx = 0
            tot_down = 0
            reg.click()
            options = browser.find_elements_by_class_name("ss-option")
            while tot_down < 20:
                flag = True
                region = options[opt_idx].text
                try:
                    options[opt_idx].click()
                except ElementNotInteractableException:
                    flag = False
                except ElementClickInterceptedException:
                    browser.execute_script("arguments[0].scrollIntoView(true);", options[opt_idx]);
                    options[opt_idx].click()
                    
                if flag:
                    time.sleep(1)
                    tot = browser.find_elements_by_class_name("ss-single-selected")[1]
                    tot.click()
                    time.sleep(1)
                    opts = browser.find_elements_by_class_name("ss-option")
                    opts[131].click()
                    time.sleep(1)
                    download.click()
                    time.sleep(1)
                    rename_file(region)
                    tot_down += 1
                    browser.refresh()
                    time.sleep(1)
                    browser.find_element_by_class_name("ss-single-selected").click()
                    options = browser.find_elements_by_class_name("ss-option")
                    download = browser.find_element_by_id("scarica-dati")
                    time.sleep(1)
                    
                opt_idx += 1
        else:
            download.click()
                
    #but if we download data we are done with the page
    else:
        #first we load all the data if a button is present
        while True:
            try:    
                load_all = browser.find_element_by_id("caricaAltri")
            except NoSuchElementException:
                break
            
            try:
                load_all.click()
            except ElementNotInteractableException:
                break 
        
        #then we indeed look for tables
        element = browser.find_element_by_xpath("//*")
        source_code = element.get_attribute("innerHTML")
        bsobj = bs(source_code, "lxml")
            
        table = bsobj.find("table")
        if table:
            rows = table.findAll("tr")
        
            names = urlparse(page).path.split("/")
            name = [name for name in names if "index" not in name][-1].strip(".html")
            name = root + name + "Table" + ".csv"
        
            #we write each table (one per page) in a csv file
            csv_file = open(name, "wt", encoding="utf-8", newline="")
            writer = csv.writer(csv_file)
        
            for row in rows:
                csv_row = []
                for cell in row.findAll(["td","th"]):
                    td = cell.find("div", {"class":"barra-testo"})
                    if td:
                        csv_row.append(bytes(td.get_text(), "utf-8").decode("ascii", "ignore").replace(".", "").replace(",","."))
                    else:    
                        csv_row.append(bytes(cell.get_text(), "utf-8").decode("ascii", "ignore").replace(".", "").replace(",","."))
                writer.writerow(csv_row)
        
            csv_file.close()

files = [root + file for file in os.listdir(root)]
data = []
data_json = []
regions = []
for file in files:
    if file.endswith(".csv"):
        if "Table" in file:
            splitter = ","
        else:
            splitter = ";"
        csv = pd.read_csv(file, sep="\n")
        columns = csv.columns.values[0].split(splitter)
        if splitter == ";":
            df = csv.iloc[:,0].str.replace("\"", "").str.replace(",", ".").str.split(splitter, expand=True)
        else:
            df = csv.iloc[:,0].str.replace("\"", "").str.split(splitter, expand=True)
        df.columns = columns
        data.append(df)
    else:
        if file.endswith(".txt"):
            json_data = open(file, "r").read()
            json_obj = json.loads(json_data)
            val = []
            for key in json_obj.keys():
                column_names = []
                column_values = []
                maxl = 0
                for country_dict in json_obj[key]:
                    column_names.append(country_dict["chiave"])
                    column = []
                    for value in country_dict["valori"]:
                        column.append(value["valore"])
                    column_values.append(column)                               
                    
                    if len(column) > maxl:
                        maxl = len(column)
                    
                for i in range(len(column_values)):
                    adder = [0 for j in range(maxl - len(column_values[i]))]
                    column_values[i] = adder + column_values[i]
                    
                df = pd.DataFrame(column_values[0])
                for i in range(1,len(column_values)):
                    df.insert(i, str(i), column_values[i])
                    
                df.columns = column_names
                data.append(df)
        else:
            regions.append(re.findall('[A-Z][^A-Z]*', file)[5])
            json_data = open(file, "r").read()
            json_obj = json.loads(json_data)        
            column_names = ["giorno"]
            columns = []
            for element in json_obj:
                column_names.append(element["name"])
                column_values = []
                days = []
                for value in element["values"]:
                    column_values.append(value["valore"])
                    days.append(value["giorno"])
                
                if days not in columns:
                    columns.append(days)
                
                columns.append(column_values)
                
            df = pd.DataFrame(columns[0])
            for i in range(1,len(columns)):
                df.insert(i, str(i), columns[i])
                                
            df.columns = column_names
            data_json.append(df)
            
data.extend(data_json)

# TODO: CREATE A FUNCTION TO PLOT THE GRAPHS WITHOUT REWRITING THE CODE EACH TIME...
# RE-READING THIS PART SEEMS LIKE I STARTED PROGRAMMING YESTERDAY

us = data[4]
us["% Terapia intensiva"] = us["terapia_intensiva"].astype(int) / us["totale_positivi"].astype(int)

us["data"] = pd.to_datetime(us["data"])
plt.plot(us["data"], us["% Terapia intensiva"]*100, color="green")
plt.plot(us["data"], us["% Terapia intensiva"]/us["% Terapia intensiva"], color="red")
plt.xticks(rotation=45, ha="right")
plt.yticks(np.arange(13))
plt.tight_layout()
plt.title('Percentuale di contagiati in terapia intensiva')
plt.xlabel('Data')
plt.ylabel('%')
plt.show()

plt.plot(us["data"][120:], us["% Terapia intensiva"][120:]*100, color="blue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Percentuale di contagiati in terapia intensiva')
plt.xlabel('Data')
plt.ylabel('%')
plt.show()

us["% morti su guariti + morti"] = us["deceduti"].astype(int) / (us["deceduti"].astype(int) + us["guariti"].astype(int))
plt.plot(us["data"][30:], us["% morti su guariti + morti"][30:]*100, color="black", label="Andamento morti/(morti + guariti)")
us["% morti su totale casi"] = us["deceduti"].astype(int) / us["totale_casi"].astype(int)
plt.plot(us["data"][30:], us["% morti su totale casi"][30:]*100, color="orange", label="Andamento morti/totale casi")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Andamento morti/(morti + guariti)')
plt.xlabel('Data')
plt.ylabel('%')
plt.legend()
plt.show()

plt.plot(us["data"], us["deceduti"].astype(int), color="black")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Andamento morti')
plt.xlabel('Data')
plt.ylabel('Morti')
plt.grid()
plt.show()
            
plt.plot(us["data"], us["totale_positivi"].astype(int), color="orange", label="Totale positivi")
plt.plot(us["data"], us["isolamento_domiciliare"].astype(int), color="green", label="Isolamento domiciliare")
plt.plot(us["data"], us["totale_positivi"].astype(int) - us["isolamento_domiciliare"].astype(int), color="blue", label="Differenza")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Andamento positivi')
plt.xlabel('Data')
plt.ylabel('Positivi')
plt.grid()
plt.legend()
plt.show()          
            
plt.plot(us["data"], us["totale_positivi"].astype(int) - us["isolamento_domiciliare"].astype(int), color="blue", label="Differenza")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Curva epidemiologica (?)')
plt.xlabel('Data')
plt.ylabel('Positivi')
plt.grid()
plt.show()          

plt.plot(us["data"], us["diff_tamponi"].astype(int), color="blue", label="Differenza")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Andamento tamponi')
plt.xlabel('Data')
plt.ylabel('Positivi')
plt.grid()
plt.show()

us = data[7] 

us["giorno"] = pd.to_datetime(us["giorno"])
plt.plot(us["giorno"], us["mediamobile_tamponi"].astype(int), color="blue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Media mobile tamponi (7giorni)')
plt.xlabel('Data')
plt.ylabel('Tamponi')
plt.grid()
plt.show()           

fig, ax1 = plt.subplots()
ax1.set_xlabel('Data')
ax1.set_ylabel('Tamponi')
l1, = ax1.plot(us["giorno"], us["mediamobile_tamponi"].astype(int), color="red", label="Tamponi totali")
ax2 = ax1.twinx()
ax2.set_ylabel('% tamponi positivi')
l2, = ax2.plot(us["giorno"], us["rapp_totale_casi_tamponi"].astype(float), color="blue", label="% tamponi positivi")
plt.legend(handles=[l1, l2])
fig.tight_layout()
fig.autofmt_xdate(rotation=45)
ax1.set_title('Media mobile tamponi (7giorni)')
plt.show()           

fig, ax1 = plt.subplots()
ax1.set_xlabel('Data')
ax1.set_ylabel('% tamponi positivi')
l1, = ax1.plot(us["giorno"][120:], us["rapp_totale_casi_tamponi"][120:].astype(float), color="blue", label="% tamponi positivi")
ax2 = ax1.twinx()
ax2.set_ylabel('Tamponi')
l2, = ax2.plot(us["giorno"][120:], us["mediamobile_tamponi"][120:].astype(int), color="red", label="Tamponi totali")
fig.tight_layout()
fig.autofmt_xdate(rotation=45)
ax1.set_title('Media mobile tamponi (7giorni)')
plt.legend(handles=[l1, l2])
plt.show()

data.append(data[3][["Luogo", "Tamponitotali", "% Contagi/tamponi"]])
us = data[-1]
us["Tamponitotali"].replace('', np.nan, inplace=True)
us.dropna(subset=['Tamponitotali'], inplace=True)
us["% tamponi sul totale"] = (us["Tamponitotali"].astype(int) / int(us["Tamponitotali"][0]))*100
us["% Contagi/tamponi"] = us["% Contagi/tamponi"].str.strip("%")

x = np.array(us["% Contagi/tamponi"][1:].astype(float)).reshape((-1, 1))
y = np.array(us["% tamponi sul totale"][1:].astype(float))

model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
intercept = model.intercept_
slope = model.coef_

x = np.linspace(0, us["% Contagi/tamponi"][1:].astype(float).max(), us["% Contagi/tamponi"][1:].size)

plt.plot(us["% Contagi/tamponi"][1:].astype(float), us["% tamponi sul totale"][1:].astype(float), "ro", color="blue")
plt.plot(x, intercept + x * slope, label= "R-squared: " + str(round(r_sq, 4) * 100) + "%")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.xlabel('% Contagi/tamponi')
plt.ylabel('% tamponi sul totale')
plt.grid()
plt.legend()
plt.show() 

us = data[5]
us["data"] = pd.to_datetime(us["data"])

ph = [0]
ph.extend([(float(us["terapia_intensiva"][i]) / float(us["terapia_intensiva"][i-1])) - 1.0 for i in range(1,us["terapia_intensiva"].size)])
us["tasso_di_crescita_TI"] = pd.Series(ph)
us["media_mobile_tasso_di_crescita_TI"] = pd.Series(moving_average(us["tasso_di_crescita_TI"], 7))

plt.plot(us["data"], us["media_mobile_tasso_di_crescita_TI"].astype(float), color="blue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title('Media mobile a 7 giorni del tasso di crescita dei ricoverati in TI')
plt.xlabel('Data')
plt.ylabel('Tasso di crescita ricoverati in terapia intensiva')
plt.grid()
plt.show()

max_d = max(us["terapia_intensiva"].astype(float))
s_1 = us["terapia_intensiva"].astype(float) / max_d

date = us["data"].copy()
ti_exp = us["terapia_intensiva"].copy()

date = date.append(pd.Series(pd.date_range(us["data"].iloc[-1], periods=25)), ignore_index=True)

p = float(us["terapia_intensiva"].iloc[-1])
for i in range(25):
    s = p * (1 + us["media_mobile_tasso_di_crescita_TI"].iloc[-1])
    ti_exp.loc[ti_exp.index.max()+1] = s 
    p = s

ti_lin = us["terapia_intensiva"].copy()
p = float(us["terapia_intensiva"].iloc[-1]) - float(us["terapia_intensiva"].iloc[-2])
for i in range(25):
    ti_lin.loc[ti_lin.index.max()+1] = p + float(ti_lin.iloc[-1])

plt.plot(date, ti_lin.astype(float), color="red")
plt.plot(date, ti_exp.astype(float), color="blue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title("Confronto tra crescita esponenziale e lineare")
plt.xlabel('Data')
plt.ylabel('Ricoverati in terapia intensiva')
plt.grid()
plt.show()

us = data[2]
us["Tasso letalit"] = us["Tasso letalit"].str.strip("%").astype(float)

i = 0
us["Flag"] = us["Tasso letalit"] * np.nan
for element in us["Paesi"]:
    if element in cnts:
        us["Flag"][i] = 1
    i += 1   

us.dropna(subset=['Flag'], inplace=True)

plt.bar(us["Paesi"],us["Tasso letalit"])
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.title("Tasso di letalità nei vari paesi Europei")
plt.xlabel('Paese')
plt.ylabel('Tasso di letalità')
plt.show()

plt.bar(us["Paesi"],us["Morti"].astype(int))
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.title("Morti totali nei vari paesi Europei")
plt.xlabel('Paese')
plt.ylabel('Morti')
plt.show()

plt.bar(us["Paesi"],us["Totali"].astype(int))
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.title("Casi totali nei vari paesi Europei")
plt.xlabel('Paese')
plt.ylabel('Casi totali')
plt.ticklabel_format(style="plain", axis="y")
plt.show()

# infected estimation
us = data[4]

us["percentuale positivi sporca"] = us["nuovi_positivi"].astype(int) / us["diff_tamponi"].astype(int)
us["percentuale positivi pulita"] = us["percentuale positivi sporca"] * (1 - abs(us["ospedalizzati"].astype(int)) / us["totale_positivi"].astype(int))
us["stima casi totali"] = us["percentuale positivi pulita"] * N
us["media mobile stima casi totali"] = pd.Series(moving_average(us["stima casi totali"], 7))

plt.plot(us["data"][98:],us["stima casi totali"][98:])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title("Stima dei casi totali in Italia")
plt.xlabel('Data')
plt.ylabel('Stima casi totali')
plt.ticklabel_format(style="plain", axis="y")
plt.show()

plt.plot(us["data"][98:],us["media mobile stima casi totali"][98:])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title("Stima dei casi totali in Italia (media mobile a 7 giorni)")
plt.xlabel('Data')
plt.ylabel('Stima casi totali')
plt.ticklabel_format(style="plain", axis="y")
plt.show()

# SIR MODEL 
# Equations:
# dS / dt = -beta * S * I / N                   S_i+1 = S_i - beta * S_i * I_i / N
# dI / dt = beta * S * I / N - gamma * I        I_i+1 = I_i + beta * S_i * I_i / N - gamma * I_i
# dR / dt = gamma * I                           R_i+1 = R_i + gamma * I_i

us["stima recovered"] = us["guariti"].astype(int) * (us["media mobile stima casi totali"].iloc[-1] / pd.Series(moving_average(us["totale_positivi"])).iloc[-1])
us["stima susceptible"] = N - us["media mobile stima casi totali"] - us["stima recovered"]

hospitality_rate = 1 / np.mean(us["media mobile stima casi totali"][220:] / us["ospedalizzati"][220:].astype(int))
nonIcu_rate = 1 / np.mean(us["media mobile stima casi totali"][220:] / us["ricoverati"][220:].astype(int))
icu_rate = 1 / np.mean(us["media mobile stima casi totali"][220:] / us["terapia_intensiva"][220:].astype(int))

sir_model_dataframe = us[["stima susceptible", "media mobile stima casi totali", "stima recovered"]][220:] / 1000
sir_model_dataframe = sir_model_dataframe.reset_index(drop=True)
sir_model_dataframe.columns = ["Susceptible", "Infected", "Recovered"]
initial_state = [sir_model_dataframe["Susceptible"][0], sir_model_dataframe["Infected"][0], sir_model_dataframe["Recovered"][0]]

msol = minimize(sumsq,[1, 0.001], args=(initial_state, sir_model_dataframe.size / 3, sir_model_dataframe["Susceptible"], sir_model_dataframe["Infected"]))
beta, gamma = msol.x

sol = solve_ivp(SIR,[0,8*sir_model_dataframe.size/3],initial_state,t_eval=np.arange(0,8*sir_model_dataframe.size/3,1), args=(beta,gamma))

plt.plot(sol.t,sol.y[0], label="Susceptible")
plt.plot(sol.t,sol.y[1], label="Infected")
plt.plot(sol.t,sol.y[2], label="Recovered")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
#plt.title("Previsione ospedalizzazioni")
plt.xlabel('Giorni')
plt.ticklabel_format(style="plain", axis="y")
plt.legend()
plt.show() 

plt.plot(sol.t,sol.y[1] * icu_rate * 1000, label="ICU", color="red")
plt.plot(sol.t[:us["terapia_intensiva"][220:].size], us["terapia_intensiva"][220:].astype(int), label="Actual ICU", color="green")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title("Previsione occupazione terapie intensive")
plt.xlabel('Giorni')
plt.ticklabel_format(style="plain", axis="y")
plt.legend()
plt.show()

us["stima infected 2"] = pd.Series(moving_average(us["totale_positivi"], 7)) * 10
us["stima recovered 2"] = us["guariti"].astype(int) * 10
us["stima susceptible 2"] = N - us["stima infected 2"] - us["stima recovered"]

hospitality_rate_2 = 1 / np.mean(us["stima infected 2"][220:] / us["ospedalizzati"][220:].astype(int))
nonIcu_rate_2 = 1 / np.mean(us["stima infected 2"][220:] / us["ricoverati"][220:].astype(int))
icu_rate_2 = 1 / np.mean(us["stima infected 2"][220:] / us["terapia_intensiva"][220:].astype(int))

sir_model_dataframe = us[["stima susceptible 2", "stima infected 2", "stima recovered 2"]][220:] / 1000
sir_model_dataframe = sir_model_dataframe.reset_index(drop=True)
sir_model_dataframe.columns = ["Susceptible", "Infected", "Recovered"]
initial_state = [sir_model_dataframe["Susceptible"][0], sir_model_dataframe["Infected"][0], sir_model_dataframe["Recovered"][0]]

msol = minimize(sumsq,[1, 0.01], args=(initial_state, sir_model_dataframe.size / 3, sir_model_dataframe["Susceptible"], sir_model_dataframe["Infected"]))
beta, gamma = msol.x

sol = solve_ivp(SIR,[0,8*sir_model_dataframe.size/3],initial_state,t_eval=np.arange(0,8*sir_model_dataframe.size/3,1), args=(beta,gamma))

plt.plot(sol.t,sol.y[0], label="Susceptible")
plt.plot(sol.t,sol.y[1], label="Infected")
plt.plot(sol.t,sol.y[2], label="Recovered")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
#plt.title("Previsione ospedalizzazioni")
plt.xlabel('Giorni')
plt.ticklabel_format(style="plain", axis="y")
plt.legend()
plt.show() 

plt.plot(sol.t,sol.y[1] * icu_rate_2 * 1000, label="ICU", color="red")
plt.plot(sol.t[:us["terapia_intensiva"][220:].size], us["terapia_intensiva"][220:].astype(int), label="Actual ICU", color="green")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.title("Previsione occupazione terapie intensive")
plt.xlabel('Giorni')
plt.ticklabel_format(style="plain", axis="y")
plt.legend()
plt.show()