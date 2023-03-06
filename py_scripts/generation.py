import random
import datetime as dt
import csv
import os
from dotenv import load_dotenv, find_dotenv
import sys

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())


last_names = {}
first_names_men = {}
first_names_women = {}
healthcare_units = []

#Get last names from last_names.csv
file_name = os.getenv('PUBLIC_DATA_DIR') + '/last_names.csv'
with open(file_name, 'r') as file:
    for line in file:
        name = line.split(';')[0].capitalize()
        number = int(line.split(';')[1])
        if(number > 100):
            last_names[name] = number

#Get first names from first_names_men.csv
file_name = os.getenv('PUBLIC_DATA_DIR') + '/first_names_men.csv'
with open(file_name, 'r') as file:
    for line in file:
        name = line.split(';')[0].capitalize()
        number = int(line.split(';')[1])
        if(number > 100):
            first_names_men[name] = number
   
file_name = os.getenv('PUBLIC_DATA_DIR') + '/first_names_women.csv'
with open(file_name, 'r') as file:
    for line in file:
        name = line.split(';')[0].capitalize()
        number = int(line.split(';')[1])
        if(number > 100):
            first_names_women[name] = number

file_name = os.getenv('PUBLIC_DATA_DIR') + '/verksamheter.csv'
with open(file_name, 'r') as file:
    for line in file:
        healthcare_units.append(line.strip())


def generate_date(dateformat="%Y%m%d"):

    #Generate a random date
    start_date = dt.date(2007, 1, 1)
    end_date = dt.date(2022, 1, 31)

    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + dt.timedelta(days=random_number_of_days)
    return random_date.strftime(dateformat)

def generate_firstname(type=None):
    if type == "man":
        return random.choices(list(first_names_men.keys()), weights=list(first_names_men.values()), k=1)[0]
    elif type == "woman":
        return random.choices(list(first_names_women.keys()), weights=list(first_names_women.values()), k=1)[0]
    else:
        #Generate a number, if > 0.5 -> man, else woman
        if random.random() > 0.5:
            return random.choices(list(first_names_men.keys()), weights=list(first_names_men.values()), k=1)[0]
        else:
            return random.choices(list(first_names_women.keys()), weights=list(first_names_women.values()), k=1)[0]

def generate_lastname():
    return random.choices(list(last_names.keys()), weights=list(last_names.values()), k=1)[0]
    
def generate_healthcareunit():
    return random.choices(healthcare_units, k=1)[0]

def generate_phonenumber():
    type = random.choices(["home", "mobile"], weights=[1,2], k=1)[0]
    if type == "home":
        start = "0"
        end = str(random.randint(1,9)) + str(random.randint(100,999)) + str(random.randint(100,999))
        return start + end
    elif type == "mobile":
        if random.random() > 0.5:
            start = "07"
            end =  str(random.randint(10,99)) + "-" +  str(random.randint(100,999)) + str(random.randint(100,999))
        else:
            start = "07"
            end = str(random.randint(1000,9999)) + str(random.randint(1000,9999))

        return start + end
    
#Generate a random age, between 18 and 100
#Have a higher chance of getting a age between 50 and 70
def generate_age(contains_string=True):
    age = random.randint(16,99)
    if age < 50 or age > 85:
        age = random.randint(16,99)

    if contains_string:
        return str(age) + "-årig"
    
    return age

def generate_city():
    return NotImplementedError

def generate_country():
    return NotImplementedError

def generate_address():
    return NotImplementedError


print(generate_age())