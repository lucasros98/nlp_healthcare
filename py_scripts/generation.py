import random
import datetime as dt
import os
from dotenv import load_dotenv, find_dotenv
import sys
import json

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

class LabelGenerator:
    def __init__(self):
        #Initialize variables
        self.last_names = {}
        self.first_names_men = {}
        self.first_names_women = {}
        self.first_names_neutral = []
        self.healthcare_units = []
        self.swedish_cities = []
        self.districts_gbg = {}
        self.countries = []
        self.addresses = []
        self.synonyms = {}

        #Blacklisted entities (entities not allowed to be generated)
        self.blacklist = {}
        self.max_tries = 100

        #Read data from csv files
        self.read_data()

        #Initiate first_names_neutral
        for name in self.first_names_men:
            if name in self.first_names_women:
                self.first_names_neutral.append(name)

    def get_gender_of_first_name(self, first_name):
        if first_name in self.first_names_neutral:
            return None
        elif first_name in self.first_names_men:
            return 'man'
        elif first_name in self.first_names_women:
            return 'woman'
        else:
            return None
        
    def get_blacklist(self,entity):
        if entity not in self.blacklist:
            return []
        else:
            return self.blacklist[entity]

    def remove_common_entities(self, list, entity):
        #Use blacklist for other entities
        self.blacklist[entity] = list

        for label in list:
            if entity == 'First_Name':
                self.first_names_women.pop(label, None)
                self.first_names_men.pop(label, None)
            elif entity == 'Last_Name':
                self.last_names.pop(label, None)
            elif entity == 'Health_Care_Unit':
                if label in self.healthcare_units:
                    self.healthcare_units.remove(label)
            elif entity == 'Location':
                if label in self.swedish_cities:
                    self.swedish_cities.remove(label)
                self.districts_gbg.pop(label, None)
                if label in self.addresses:
                    self.addresses.remove(label)
                if label in self.countries:
                    self.countries.remove(label)

    def read_data(self):
        #Get last names from last_names.csv
        file_name = os.getenv('PUBLIC_DATA_DIR') + '/last_names.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(';')[0].capitalize()
                number = int(line.split(';')[1])
                if(number > 300):
                    self.last_names[name] = number

        #Get first names from first_names_men.csv
        file_name = os.getenv('PUBLIC_DATA_DIR') + '/first_names_men.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(';')[0].capitalize()
                number = int(line.split(';')[1])
                if(number > 500):
                    self.first_names_men[name] = number
        
        file_name = os.getenv('PUBLIC_DATA_DIR') + '/first_names_women.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(';')[0].capitalize()
                number = int(line.split(';')[1])
                if(number > 500):
                    self.first_names_women[name] = number

        file_name = os.getenv('PUBLIC_DATA_DIR') + '/gbg_districts.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(',')[0].capitalize()
                number = float(line.split(',')[1])
                self.districts_gbg[name] = number

        file_name = os.getenv('PUBLIC_DATA_DIR') + '/verksamheter.csv'
        with open(file_name, 'r') as file:
            for line in file:
                self.healthcare_units.append(line.strip())

        file_name = os.getenv('PUBLIC_DATA_DIR') + '/addresses.csv'
        with open(file_name, 'r') as file:
            for line in file:
                self.addresses.append(line.strip())

        file_name = os.getenv('PUBLIC_DATA_DIR') + '/swedish_cities.csv'
        with open(file_name, 'r') as file:
            for line in file:
                self.swedish_cities.append(line.strip())

        file_name = os.getenv('PUBLIC_DATA_DIR') + '/countries.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(',')[0].capitalize()
                self.countries.append(name)
        
        file_name = os.getenv('PUBLIC_DATA_DIR') + '/synonyms.json'
        with open(file_name, 'r') as file:
            self.synonyms = json.load(file)

    def generate_random_entity(self, entity,params={"gender":None}):
        if entity == 'First_Name':
            if params['gender']:
                return self.generate_firstname(gender=params['gender'])
            else: 
                return self.generate_firstname()
        elif entity == 'Last_Name':
            return self.generate_lastname()
        elif entity == 'Health_Care_Unit':
            return self.generate_healthcare_unit()
        elif entity == 'Location':
            return self.generate_location()
        elif entity == 'Full_Date':
            return self.generate_date()
        elif entity == 'Date_Part':
            return self.generate_datepart()
        elif entity == 'Age':
            return self.generate_age()
        elif entity == 'Phone_Number':
            return self.generate_phonenumber()
        else:
            return None
    
    def generate_date(self,dateformat="%Y%m%d"):
        blacklisted_dates = self.get_blacklist('Date')

        #Generate a random date
        start_date = dt.date(2007, 1, 1)
        end_date = dt.date(2022, 1, 31)

        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days

        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + dt.timedelta(days=random_number_of_days)

        #Check if date is blacklisted
        max_tries = self.max_tries
        while random_date.strftime(dateformat) in blacklisted_dates and max_tries > 0:
            random_number_of_days = random.randrange(days_between_dates)
            random_date = start_date + dt.timedelta(days=random_number_of_days)
            max_tries -= 1

        return random_date.strftime(dateformat)

    def generate_datepart(self, dateformat="%d/%m"):
        blacklisted_dates = self.get_blacklist('Date_Part')
        date = self.generate_date(dateformat)

        max_tries = self.max_tries
        while date in blacklisted_dates and max_tries > 0:
            date = self.generate_date(dateformat)
            max_tries -= 1

        return date

    def generate_firstname(self,gender=None):
        if gender == "man":
            return random.choices(list(self.first_names_men.keys()), weights=list(self.first_names_men.values()), k=1)[0]
        elif gender == "woman":
            return random.choices(list(self.first_names_women.keys()), weights=list(self.first_names_women.values()), k=1)[0]
        else:
            #Generate a number, if > 0.5 -> man, else woman
            if random.random() > 0.5:
                return random.choices(list(self.first_names_men.keys()), weights=list(self.first_names_men.values()), k=1)[0]
            else:
                return random.choices(list(self.first_names_women.keys()), weights=list(self.first_names_women.values()), k=1)[0]

    def generate_lastname(self):
        return random.choices(list(self.last_names.keys()), weights=list(self.last_names.values()), k=1)[0]
        
    def generate_healthcare_unit(self):
        return random.choices(self.healthcare_units, k=1)[0]

    def generate_number(self):
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

    def generate_phonenumber(self):
        blacklisted_phonenumbers = self.get_blacklist('Phone_Number')

        #Generate a random phonenumber
        number = self.generate_number()

        max_tries = self.max_tries
        while number in blacklisted_phonenumbers and max_tries > 0:
            number = self.generate_number()
            max_tries -= 1

        return number
        
        
    def generate_age(self,contains_string=None):
        age = random.randint(16,99)
        if age < 50 or age > 85:
            age = random.randint(16,99)

        rand = random.random()
        if contains_string or rand < 0.5:
            return str(age) + "-årig"
        
        return age

    def generate_city(self):
        return random.choices(self.swedish_cities, k=1)[0]

    def generate_country(self):
        return random.choices(self.countries, k=1)[0]

    def generate_address(self):
        return random.choices(self.addresses, k=1)[0]

    def generate_district(self):
        return random.choices(list(self.districts_gbg.keys()), weights=list(self.districts_gbg.values()), k=1)[0]

    def generate_location(self):
        rand = random.random()
        if rand < 0.3:
            return self.generate_city()
        elif rand < 0.6:
            return self.generate_district()
        elif rand < 0.9:
            return self.generate_address()
        else:
            return self.generate_country()

    def generate_synonym(self, word):
        chosen_synonym = word
        for syn in self.synonyms:
            if syn['word'] == word:
                chosen_synonym = random.choices(syn['synonyms'], k=1)[0]
        return chosen_synonym