#!/usr/bin/env python
# coding: utf-8

# In[1]:


name = input("Enter your name: ")

weight = int(input("Enter your weight in pounds: "))

height = int(input("Enter your wheight in inches: "))

BMI = (weight * 703)/ (height * height)

print(BMI)

if BMI > 0:
    if(BMI<18.5):
        print(name + ", You are underweight.")
    elif(BMI<=24.9):
        print(name + ", You are normal weight.")
    elif(BMI<29.9):
        print(name + ", You are overweight.")
    elif(BMI<34.9):
        print(name + ", You are obese.")
    elif(BMI<39.9):
        print(name + ", You are severly obese.")
    else:
        print(", You are morbidly obese.")
else:
        print(", Enter valid input")


# In[ ]:



