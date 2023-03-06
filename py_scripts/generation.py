import random
import datetime as dt
import os
from dotenv import load_dotenv, find_dotenv
import sys

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

class LabelGenerator:
    def __init__(self):
        #Initialize variables
        self.last_names = {}
        self.first_names_men = {}
        self.first_names_women = {}
        self.healthcare_units = []
        self.swedish_cities = []
        self.districts_gbg = {}
        self.countries = []
        self.addresses = []

        #Read data from csv files
        self.read_data()

    def remove_common_entities(self, list, entity):
        for label in list:
            if entity == 'First_Name':
                self.first_names_women.pop(label, None)
                self.first_names_men.pop(label, None)
            elif entity == 'Last_Name':
                self.last_names.pop(label, None)
            elif entity == 'Health_Care_Unit':
                self.healthcare_units.remove(label)
            elif entity == 'Location':
                self.swedish_cities.remove(label)
                self.districts_gbg.pop(label, None)
                self.addresses.remove(label)
                self.countries.remove(label)

    def read_data(self):
        #Get last names from last_names.csv
        file_name = os.getenv('PUBLIC_DATA_DIR') + '/last_names.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(';')[0].capitalize()
                number = int(line.split(';')[1])
                if(number > 100):
                    self.last_names[name] = number

        #Get first names from first_names_men.csv
        file_name = os.getenv('PUBLIC_DATA_DIR') + '/first_names_men.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(';')[0].capitalize()
                number = int(line.split(';')[1])
                if(number > 100):
                    self.first_names_men[name] = number
        
        file_name = os.getenv('PUBLIC_DATA_DIR') + '/first_names_women.csv'
        with open(file_name, 'r') as file:
            for line in file:
                name = line.split(';')[0].capitalize()
                number = int(line.split(';')[1])
                if(number > 100):
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

    def generate_date(self,dateformat="%Y%m%d"):

        #Generate a random date
        start_date = dt.date(2007, 1, 1)
        end_date = dt.date(2022, 1, 31)

        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days

        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + dt.timedelta(days=random_number_of_days)
        return random_date.strftime(dateformat)

    def generate_datepart(self, dateformat="%d/%m"):
        return self.generate_date(dateformat)

    def generate_firstname(self,type=None):
        if type == "man":
            return random.choices(list(self.first_names_men.keys()), weights=list(self.first_names_men.values()), k=1)[0]
        elif type == "woman":
            return random.choices(list(self.first_names_women.keys()), weights=list(self.first_names_women.values()), k=1)[0]
        else:
            #Generate a number, if > 0.5 -> man, else woman
            if random.random() > 0.5:
                return random.choices(list(self.first_names_men.keys()), weights=list(self.first_names_men.values()), k=1)[0]
            else:
                return random.choices(list(self.first_names_women.keys()), weights=list(self.first_names_women.values()), k=1)[0]

    def generate_lastname(self):
        return random.choices(list(self.last_names.keys()), weights=list(self.last_names.values()), k=1)[0]
        
    def generate_healthcareunit(self):
        return random.choices(self.healthcare_units, k=1)[0]

    def generate_phonenumber(self):
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
        
    def generate_age(self,contains_string=True):
        age = random.randint(16,99)
        if age < 50 or age > 85:
            age = random.randint(16,99)

        if contains_string:
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