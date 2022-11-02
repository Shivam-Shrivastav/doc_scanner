import streamlit as st
import pickle
import requests
from streamlit_lottie import st_lottie
import bz2file as bz2

import cv2
import numpy as np
from imutils.perspective import four_point_transform

from PIL import Image, ImageEnhance
from IPython.display import display
import torchvision.transforms.functional as F
# import img2pdf
import os
# from fpdf import FPDF
import pytesseract
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline




pytesseract.pytesseract.tesseract_cmd =  r'tesseract/5.2.0/bin/tesseract'
count = 0
scale = 0.5

font = cv2.FONT_HERSHEY_SIMPLEX

WIDTH, HEIGHT = 1920, 1080

@st.cache
def image_processing(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = Image.fromarray(image)
    image = F.adjust_contrast(image, 1.1)
    image = F.adjust_brightness(image, 1.1)
    image = F.adjust_sharpness(image, 8)  
    image = F.adjust_contrast(image, 1.2)
    image = F.adjust_brightness(image, 1.1)
    image = F.adjust_contrast(image, 1.2)
    image = F.adjust_brightness(image, 1.1)
    image = F.adjust_contrast(image, 1.2)
    image = F.adjust_brightness(image, 1.1)
    image = np.array(image)
    image = cv2.fastNlMeansDenoisingColored(image,None, 10, 10, 7, 21)
    
    image = np.array(image)

    return image


@st.cache
def edge_detection(image):
    global document_contour

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 3)
    #display(Image.fromarray(image))


@st.cache
def doc_scan(image):
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    print(frame)
    #display(Image.fromarray(frame))
    frame_copy = frame
    # display(Image.fromarray(frame_copy))
    
    edge_detection(frame_copy)
    # display(Image.fromarray(frame_copy))
    

#     cv2.imshow("input", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))

    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    
#     cv2.imshow("Warped", cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0]))))



    processed = image_processing(warped)
    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    processed = processed[10:processed.shape[0] - 30, 10:processed.shape[1] - 30]
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(processed))
    processed = Image.fromarray(processed)
    return processed

@st.cache
def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

@st.cache
def display_summary(model, article):
        with open(model, 'rb') as pickle_file:
            summarizer = bz2.BZ2File(pickle_file)
            summarizer = pickle.load(summarizer)
        summary = summarizer(article, max_length=200, min_length=100, do_sample=False)[0]['summary_text']
        if summary:
            return summary

with open('qamodel.pkl', 'rb') as pickle_file:
        qamodel = pickle.load(pickle_file)

model_name = qamodel
# @st.cache
# def qa(qamodel):
    

#     return model_name

    







st.set_page_config(page_title= "Smart Doc Scanner", page_icon= ":fax:", layout= "wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_4kmk2efh.json")


# HEADER SECTION:
with st.container():
    left, right = st.columns(2)
    with left:
        st.title("Smart Document Scanner :fax:")
        st.write("""This app scan the physical documents and make it digitalized in pdf format and give us the brief summary of the whole article.""")
        st.write("We can also do personalized question answering over the article in the document.")
        st.write("[Learn about creator Shivam Shrivastava > ](https://shivam-shrivastav.github.io)")
    with right:
        st_lottie(lottie_coding, height= 400)




st.write("##")


st.subheader("Upload the Document Image you want to scan and get summary of it")

summary = None
image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image_file is not None:
    # To read file as bytes:
    bytes_data = image_file.getvalue()
    # st.write(type(image_file))
    # st.write(dir(image_file))
    col1, col2 = st.columns(2)
    col1.subheader("Original Image")
    col1.image(image_file, width=500)
    col1.write(type(image_file))
    # col1.write(image_file.getvalue())
    col2.subheader("Scanned Image")
    col2.image(doc_scan(bytes_data), width = 500)
    st.write("##")
    st.write("##")
    st.write("##")

    article = ocr_core(doc_scan(bytes_data))


    st.subheader("Summary of the Article :page_with_curl::")
    st.write(display_summary('summarizer.pbz2', article))
    
    st.write("##")
    st.write("##")
    st.write("##")

    st.subheader("Article related question answering :question::question::")
    # model_name = qa()
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    title = st.text_input('Enter the question you want to ask related to article', "")

    if title != '':
        QA_input = {
            'question': title,
            'context': article
        }
        res = nlp(QA_input)

        # b) Load model & tokenizer
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)


        st.write("Answer :trophy: : ", res['answer'])

    

    









