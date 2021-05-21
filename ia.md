# Inteligencia Artificial: Ejemplo de Red Neuronal en Python

## Descripción
Las Redes Neuronales artificiales son uno de los modelos computacionales mas utilizados para implementar Inteligencia Artificial. Consisten en un conjunto de unidades, llamadas neuronas, conectadas entre sí para transmitir señales.

Cada neurona se encuentra conectada con otras neuronas por enlaces. A traves de ellos, recibe información de otras neuronas, la evalúa, y as su vez informa a otras neuronas del resultado de esa evaluación, que puede ser tanto para resaltar una caracteristica como para atenuarla.

## Objetivo
Crearemos una Red Neuronal con la que examinaremos automóviles y predeciremos la distancia que puede recorrer por cada galón de combustible.

## Estructura
Comunmente las Redes Neuronales se implementan en forma de Capas. La primer capa, llamada capa de entrada o capa 0, representa las distintas caracteristicas que se evalúan de un objeto.

En nuestro ejemplo, la primer capa va a estar representada por las características que evaluaremos del automovil. Por ejemplo: La cantidad de cilindros, el Peso, la Aceleración, el Año y el Origen de fabricación, etc.

Luego agregaremos dos capas más, llamadas ocultas, que serán las encargadas de evaluar la información, la relación entre las distintas características y emitir un resultado a la capa posterior.

Finalmente tendremos una Capa de Salida, que obtendra los resultados de capa oculta anterior y calculará la distancia que el automóvil puede recorrer.

## Las neuronas
Cada capa va a estar formada por varias neuronas, excepto la última capa, la capa de salida, que contará con solo una neurona.

Cada neurona sera representada por la siguiente fórmula:

$$
\Large
n = \sum W X + b
$$

En donde $\Large X$ es un vector (matriz unidimensional) con la información proveniente de cada una las neuronas de la capa anterior. En el caso de la primer capa oculta, la informacion que recibirá sera directamente las características del automóvil.

$\Large W$ es un vector que le asignará un peso a cada valor proveniente de las neuronas de la capa anterior. Estos pesos serán los que generarán una respuesta que tienda a conseguir la prediccion final.

$\Large b$ es un valor que aplicará un dezplazamiento (bias) al resultado.

## Activación no-lineal
La función anterior es una función lineal. Si todas las neurona fueran lineales, el resultado de la Red Neuronal también sería otra funcion lineal. Esto no permitiría que la Red identificara condiciones complejas.

Para evitar esto, se agrega un componente no-lineal a cada neurona, llamada función de activación. En nuestro caso utilizaremos la tangente hiperbólica.

$$\Large a = \tanh( n )$$

De manera que la formula completa de cada neurona, exceptuando a la de la capa de Salida, será:

$$
\Large
a = tanh( \sum W X + b )
$$

## Activación lineal final
Como nuestro objetivo es predecir un valor (regresion), en lugar detectar o clasificar un objeto (clasificación), la última neurona de la red, la de la capa de salida, deberá generar un valor lineal.

La fórmula de esta neurona quedará como:

$$
\Large
a = \sum W X + b
$$


