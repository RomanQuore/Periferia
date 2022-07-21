from tkinter import *
import time
import json
from flask import  Flask, request
import requests
import nltk #lenguaje natural
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy 
import tensorflow
import tflearn
import random
import pandas as pd
import numpy as np
import markdown
import time
import pickle
from pandas.io.json import json_normalize
import subprocess
from pydub import audio_segment
import speech_recognition as sr


bot_name = "Talkie"

#colores
BG_GRAY = "#EEDFCC"
BG_COLOR = "#BBFFFF"
TEXT_COLOR = "#292421"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

opcion1= []
correo= []
telefono=[]
pais=[]
empresa=[]
opcion2=[]



def _init_():
	ventana = Tk()
	# _setup_main_ventana()

	ventana.title("ChatBot")
	ventana.resizable(width=False, height=False)
	ventana.configure(width=470, height= 550, bg=BG_COLOR)
	

	head_label = Label(ventana, bg=BG_COLOR, fg=TEXT_COLOR, text="Periferia It Group", 
						font=FONT_BOLD, pady=10)

	head_label.place(relwidth=1)

	#divisores

	line = Label(ventana, width=450, bg=BG_GRAY)
	line.place(relwidth=1, rely=0.07, relheight=0.012)

	#text widget
	global text_widget

	text_widget = Text(ventana, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
						font=FONT, padx=5, pady=5)

	text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
	text_widget.configure(cursor="arrow", state=DISABLED)

	#Barra de desplazamiento 
	barra_desp = Scrollbar(text_widget)
	barra_desp.place(relheight=1, relx=0.974)
	barra_desp.configure(command=text_widget.yview)

	#boton

	boton_label = Label(ventana, bg=BG_GRAY, height=80)
	boton_label.place(relwidth=1, rely=0.825)

	#entrada de mensaje
	global msg_entrada
	msg_entrada = Entry(boton_label, bg="#96CDCD", fg=TEXT_COLOR, font=FONT)
	msg_entrada.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
	msg_entrada.focus()
	msg_entrada.bind("<Return>", on_enter_pressed()) #_on_enter_pressed


	#boton envio
	boton_envio =Button(boton_label, text="enviar", font=FONT_BOLD, width=20, bg="#96CDCD",
						command=lambda: on_enter_pressed())
	boton_envio.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

	respuesta_principal = "Hola, yo soy Talkie, soy el bot de Periferia IT Group. Es un gusto para mí atenderte\nMe gustaría conocerte más ¿Cuál es tu nombre?"
	opcion1.append("humano")

	msg3 = f"{bot_name}:{respuesta_principal}\n\n"
	text_widget.configure(state=NORMAL)
	text_widget.insert(END, msg3)
	text_widget.configure(state=DISABLED)





	ventana.mainloop()


def on_enter_pressed():


	global  msg



	msg = msg_entrada.get()

	print(msg,"este es el mensaje del cliente")
	if not msg:
		return
	nombre =opcion1[0]
	print(nombre, "kkkkkkkkkkkkkkkkkkkkkkkkkk")
	if nombre == "humano":
		print("si entro ")
		nombre = msg
		print("el nombre es:" , nombre)
	else:
		print("no hay")


	msg_entrada.delete(0,END)
	msg1 = f"{'Tu'}:{msg}\n\n"
	text_widget.configure(state=NORMAL)
	text_widget.insert(END, msg1)
	text_widget.configure(state=DISABLED)


	# #variables
	# opcion1= []
	print(opcion1,"esta es mi opcion")
	if len(opcion1) < 1:
		msgs = msg

		if len(correo) < 1:
			msgs = msg

			if len(telefono) < 1:
				msgs = msg

				if len(pais) < 1:
					msgs = msg

					if len(empresa) < 1:
						msgs = msg

						if len(opcion2) < 1:
							msgs = msg
						else:
							res = opcion2[-1]
							msgs = (res + msg)
					else:
						msgs = empresa[-1]

				else:
					msgs = pais[-1]

			else:
				msgs = telefono[-1]

		else:
			msgs = correo[-1]
	else:
		msgs = opcion1[-1]



	print(msgs, "este es el mensaje")


	with open("entrenamiento.json", 'r') as archivo:
		datos = json.load(archivo)
	try:
		with open("variables.pickle", "rb") as archivoPickle:
			palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
	except:
		palabras=[]
		tags=[]
		auxX=[] #auxiliares 
		auxY=[]
		for contenido in datos["contenido"]:
			for patrones in contenido["patrones"]: #acceder a cualquier elemento
				auxPalabra = nltk.word_tokenize(patrones) #separar palabras Reconocer puntos especiales
				palabras.extend(auxPalabra)
				auxX.append(auxPalabra)
				auxY.append(contenido["tag"])
				if contenido["tag"] not in tags:	
					tags.append(contenido["tag"])
	#Entranamiento aprendizaje automatico
		palabras = [stemmer.stem(w.lower())for w in palabras if w!="?"]#pasar todas las palabras en minuscular
		palabras = sorted(list(set(palabras)))
		tags = sorted(tags)
		entrenamiento = []
		salida=[]
		salidaVacia = [0 for _ in range(len(tags))]
		for x, documento in enumerate(auxX):
			cubeta=[]
			auxPalabra=[stemmer.stem(w.lower()) for w in documento]
			for w in palabras:
				if w in auxPalabra:
					cubeta.append(1)	
				else:
					cubeta.append(0)
			filaSalida = salidaVacia[:]	
			filaSalida[tags.index(auxY[x])]=1
			entrenamiento.append(cubeta)
			salida.append(filaSalida)
	#Definición redes neuronales a utilizar
		entrenamiento = numpy.array(entrenamiento)
		salida = numpy.array(salida)

		with open ("variables.Pickle", "wb") as archivoPickle:
			pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)
	
	tensorflow.reset_default_graph()
	red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
	red = tflearn.fully_connected(red,100)
	red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")
	red = tflearn.regression(red) #probabilidades
	modelo = tflearn.DNN(red)
	try: 
		modelo.load("modelo.tflearn")
	except:
		modelo.fit(entrenamiento,salida,n_epoch=1000,batch_size=100,show_metric=True) #bactch_size es la cantidad de neuronas en la red 
		modelo.save("modelo.tflearn")

	entrada = msgs
	print(entrada, "verificar")
	cubeta = [0 for _ in range(len(palabras))]
	entradaProcesada = nltk.word_tokenize(entrada)
	entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
	for palabraIndividual in entradaProcesada:
		for i,palabra in enumerate(palabras):
			if palabra == palabraIndividual:
				cubeta[i]=1
	resultados =modelo.predict([numpy.array(cubeta)])
	resultadosIndices = numpy.argmax(resultados)
	global tag
	tag = tags[resultadosIndices]
	print(tag, "---------------------------------------")

	opcion1.clear()
	correo.clear()
	telefono.clear()
	pais.clear()
	empresa.clear()
	opcion2.clear()

	if "1" == tag:
		respuesta = "Me gustaría conocerte más, ¿cuál es tu nombre?"
		opcion1.append("humano")


	elif "TalentoIT" == tag:
		respuesta = (" " + str(nombre) + "en Periferia IT Group conformamos en menos de 5 días el suministro de talento humano especializado en las últimas tendencias tecnológicas."
					"\n¿Necesitas talento humano especializado en TI? \n\n Si \n No")

	# elif "2" == tag:
	# 	respuesta = "Me gustaría conocerte más, ¿cuál es tu nombre?"
	# 	opcion1.append("autom")
					
	# elif "Automatizacion" == tag:
	# 	respuesta = ("en Periferia IT Group contamos con la tecnología, el equipo de trabajo y expertos para realizar las automatizaciones adecuadas para tu negocio y tus productos."
	# 				"\n¿Te gustaría agendar una cita para revisar a detalle tu requerimiento de Pruebas?\n\n Si \n No")

	# elif "3" == tag:
	# 	respuesta = "Me gustaría conocerte más, ¿cuál es tu nombre?"
	# 	opcion1.append("opcion soft")

	# elif "Desarrollo" == tag:
	# 	respuesta = ("contamos con la tecnología y las herramientas necesarias para desarrollar tus requerimientos en el menor tiempo posible ."
	# 				"\n¿Te gustaría agendar una cita para revisar a detalle tu requerimiento de Desarrollo Software?\n\n Si \n No")

	# elif "4" == tag:
	# 	respuesta = "Me gustaría conocerte más, ¿cuál es tu nombre?"
	# 	opcion1.append("trab nosotros")

	# elif "Trabaja" == tag:
	# 	respuesta = ("veo que estás interesad@ en formar parte de nuestra comunidad Periferia IT Group.\n¿Elige una de estas opciones?"
	# 				"\n\n1.Ver oferta laboral. \n2.Enviar hoja de vida")

	# 	opcion2.append("1")


	elif "oferta" == tag:
		respuesta= ("Tenemos las siguientes oportunidades laborales que encontrarás en el siguiente link."
					"\n\nVer oportunidades (Link va dentro frase https://www.facebook.com/periferiaitgroup/jobs/?ref=page_internal)"
					"\n\nSi te apasiona la tecnología, las ganas de aprender y el mejoramiento continuo, este es tu lugar."
					"Envíanos tu hoja de vida al correo: luzortiz@cbit-online.com\n\nNo olvides indicar en el asunto la tecnología en la que tienes experiencia."
					"¡Fue un placer atenderte!\nTe invitamos a seguirnos en nuestras redes sociales y mantenerte conectado con nuestra comunidad Periferia IT Group")
	
	elif "hojavida" == tag:
		respuesta= ("Si te apasiona la tecnología, las ganas de aprender y el mejoramiento continuo, este es tu lugar."
					"Envíanos tu hoja de vida al correo: luzortiz@cbit-online.com\n\nNo olvides indicar en el asunto la tecnología en la que tienes experiencia."
					"¡Fue un placer atenderte!\nTe invitamos a seguirnos en nuestras redes sociales y mantenerte conectado con nuestra comunidad Periferia IT Group")


	elif "5" == tag:
		respuesta = "Me gustaría conocerte más, ¿cuál es tu nombre?"
		opcion1.append("cuentanos")

	elif "contactenos" == tag:
		respuesta: "¡Qué bien!\ncuéntanos un poco más de lo que necesitas"
		info.append("informacion")


	elif "respCliente" == tag:
		respuesta = ("¡Perfecto!\nTe pondremos en contacto con uno de nuestros expertos en TI para atender tu requerimiento a la mayor brevedad."
					"\nPor favor escribe tu correo electrónico")


	elif "si" == tag:
		respuesta = "¡Perfecto! Es el momento de llevar tus proyectos a otro nivel. \n\nPor favor escribe tu correo electrónico"
		correo.append("correo")


	elif "no" == tag:
		respuesta = ("No te preocupes! podemos proporcionarte otros servicios de tecnología que llevarán tus proyectos al siguiente nivel"
					"\n\nPuedes revisar nuestro portafolio de servicios aquí: \nPortafolio de servicios Periferia IT Group."
					"(\nlink va dentro de frase https://periferiaitgroup.com/brochure-2020.pdf)")
	
	elif "correo" == tag:
		respuesta = "Ahora, escribe tu número de teléfono"
		telefono.append("telefono")

	elif "telefono" == tag:
		respuesta = "¿En qué país te encuentras?"
		pais.append("pais")

	elif "pais" == tag:
		respuesta = "¿De qué empresa nos escribes?"
		empresa.append("empresa")

	elif "empresa" == tag:
		respuesta = ("Muchas gracias por darnos tu información. Te pondremos en contacto con uno de nuestros expertos en TI para atender tu requerimiento a la mayor brevedad."
					"¡Fue un placer atenderte! \na continuación puedes revisar nuestro portafolio de servicios aquí:"
					"Portafolio de servicios Periferia IT Group.\n(link va dentro de frase https://periferiaitgroup.com/brochure-2020.pdf)")





	elif "Adios" == tag:
		respuesta=""




	msg2 = f"{bot_name}:{respuesta}\n\n" #{get_response(msg)}\n\n"
	text_widget.configure(state=NORMAL)
	text_widget.insert(END, msg2)
	text_widget.configure(state=DISABLED)

	text_widget.see(END)


	# ventana.mainloop()





if __name__ == "__main__":
	app = _init_()
	app.run()