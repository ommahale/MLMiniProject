from selenium import webdriver
from selenium.webdriver.common.by import By
import time
# driver  =  webdriver.Chrome()
# driver.get('http://localhost:5173')
# print(driver.title)
text= 'Selenium supports automation of all the major browsers in the market through the use of WebDriver. WebDriver is an API and protocol that defines a language-neutral interface for controlling the behaviour of web browsers. Each browser is backed by a specific WebDriver implementation, called a driver. The driver is the component responsible for delegating down to the browser, and handles communication to and from Selenium and the browser. This separation is part of a conscious effort to have browser vendors take responsibility for the implementation for their browsers. Selenium makes use of these third party drivers where possible, but also provides its own drivers maintained by the project for the cases when this is not a reality. The Selenium framework ties all of these pieces together through a user-facing interface that enables the different browser backends to be used transparently, enabling cross-browser and cross-platform automation. Selenium setup is quite different from the setup of other commercial tools. Before you can start writing Selenium code, you have to install the language bindings libraries for your language of choice, the browser you want to use, and the driver for that browser.'
# text_area=driver.find_element(By.ID, value='input_text').send_keys(text)
# button = driver.find_element(By.TAG_NAME, value = 'button')
# button.click()
# time.sleep(20)
# output = driver.find_element(by=By.ID, value='output_text').get_property('value')
# if output is None:
#     print('Test Passed')


#=============================================================================================================================
class SeleniumTestSummary:

    def __init__(self,url:str) -> None:
        self.url=url
        self.driver = webdriver.Chrome()
        self.text_area=None
        self.output=None
        self.button=None


    def test_1(self):
        print('==================== ✅Test 1 ========================')
        print(' * Server connectivity ')
        try:
            self.driver.get(self.url)
            print('Test passed  ')
        except:
            print('❌Test Failed')

    def test_2(self,id:str):
        print('==================== ✅Test 2 ========================')
        print(' * Element availability ')
        try:
            self.text_area=self.driver.find_element(By.ID, value=id)
            print('Test passed  ')
        except:
            print('❌Test Failed')
    
    def test_3(self,input:str):
        print('==================== ✅Test 3 ========================')
        print(' * Input Test ')
        try:
            self.text_area.send_keys(input)
            print('Test passed  ')
        except:
            print('❌Test Failed')

    def test_4(self):
        print('==================== ✅Test 4 ========================')
        print(' * Request Initiation ')

        try:
            self.button= self.driver.find_element(By.TAG_NAME, value = 'button')
            self.button.click()
            time.sleep(7)
            print('Test passed  ')
        except:
            print('❌Test Failed')


    
    def test_5(self,otuput_tag:str):
        print('==================== ✅Test 5 ========================')
        print(' * Response Finalize Test ')

        try:
            output_text = self.driver.find_element(by=By.ID, value=otuput_tag).get_property('value')
            if output_text == '':
                print('xxxxxxxxxxxxxxxxx ❌Test Failed : (Recieved no response from backend) xxxxxxxxxxxxxxxxx')
            else:
                print('Test passed  ')
        except:
            print('❌Test Failed')



#=============================================================================================================================

if __name__ =='__main__':
    tester= SeleniumTestSummary('http://localhost:5173')
    tester.test_1()
    tester.test_2('input_text')
    tester.test_3(text)
    tester.test_4()
    tester.test_5('output_text')