# Inteligencia Artificial: Ejemplo de Red Neuronal en Python

## Descripción
Las Redes Neuronales artificiales son uno de los modelos computacionales mas utilizados para implementar Inteligencia Artificial. Consisten en un conjunto de unidades, llamadas neuronas, conectadas entre sí para transmitir señales.

Cada neurona se encuentra conectada con otras neuronas por enlaces. A traves de ellos, recibe información de otras neuronas, la evalúa, y as su vez informa a otras neuronas del resultado de esa evaluación, que puede ser tanto para resaltar una caracteristica como para atenuarla.

## Objetivo
Crearemos una Red Neuronal con la que examinaremos automóviles y predeciremos la distancia que puede recorrer por cada galón de combustible.

Primero se explicarán algunos conceptos y lue

## Estructura
Comunmente las Redes Neuronales se implementan en forma de Capas. La primer capa, llamada capa de entrada o capa 0, representa las distintas caracteristicas que se evalúan de un objeto.

En nuestro ejemplo, la primer capa (también llamada capa 0) va a estar representada por las características que evaluaremos del automovil. Por ejemplo: La cantidad de cilindros, el Peso, la Aceleración, el Año y el Origen de fabricación, etc.

Luego agregaremos dos capas más, llamadas ocultas, que serán las encargadas de evaluar la información, la relación entre las distintas características y emitir un resultado a la capa posterior.

Finalmente tendremos una Capa de Salida, que obtendra los resultados de capa oculta anterior y calculará la distancia que el automóvil puede recorrer.

## Las Neuronas
Cada capa va a estar formada por varias neuronas, excepto la última capa, la capa de salida, que contará con solo una neurona.

Cada neurona sera representada por la siguiente fórmula:

$$
\Large
Z = \sum W X + b
$$

En donde $\Large X$ es un vector (matriz unidimensional) con la información proveniente de cada una las neuronas de la capa anterior. En el caso de la primer capa oculta, la informacion que recibirá sera directamente las características del automóvil.

$\Large W$ es un vector que le asignará un peso a cada valor proveniente de las neuronas de la capa anterior. Estos pesos serán los que generarán una respuesta que tienda a conseguir la prediccion final.

$\Large b$ es un valor que aplicará un dezplazamiento (bias) al resultado.

## Activación no-lineal
La función anterior es una función lineal. Si todas las neurona fueran funciones lineales, el resultado de la Red Neuronal también sería otra funcion lineal. Esto NO permitiría que la Red identificara relaciones complejas.

Para evitar esto, se agrega un componente no-lineal a cada neurona, llamada función de activación. En nuestro caso utilizaremos la función ReLU (unidad lineal rectificada), que no es otra cosa que:

$$\Large A = ReLU( Z ) = max( 0, Z )$$

De manera que la formula completa de cada neurona, exceptuando a la de la capa de Salida, será:

$$\Large A = max( 0, \sum W X + b )$$

## Activación lineal final
Como nuestro objetivo es predecir un valor (regresion), en lugar detectar o clasificar un objeto (clasificación), la última neurona de la red, la de la capa de salida, deberá generar un valor lineal.

La fórmula de esta neurona quedará como:

$$\Large A = \sum W X + b$$

## Optimizacion de la Red Neuronal: Aprendizaje

El proceso de aprendizaje de la Red implicará buscar valores para los pesos que usamos, de manera que cuando se la alimenta con los datos de un automóvil, la red produzca el resultado que queremos estimar: el recorrido por galon.

**Pasos:**

1. Inicializar los pesos que usaremos en nuestra red de forma aleatoria y con una distribución que facilite los calculos.
2. Repetir lo siguiente:
    1. Alimentar la red con las características de un automóvil.
    2. Realizar los cálculos de la Red Neuronal, multiplicando los pesos por los valores de entrada, sumando los desplazamientos y aplicando las funciones de activación capa por capa, hasta obtener el resultado final.
    3. Calcular el error (diferencia) entre la estimación de la Red y el valor real, que llamaremos $\Large J(W, b)$
    4. Desde atrás hacia adelante, utilizando la diferencia obtenida, calculamos las derivadas para los pesos $\Large W$ y el desfazaje $\Large b$ empleando las siguientes fórmulas:

    $$\Large \frac{\partial J(W, b)}{\partial W}  y  \frac{\partial J(W, b)}{\partial b}$$

    5. Con las derivadas, procedemos a modificar los pesos para acercarlos al punto en que la diferencia alcance un mínimo. Para ello multiplicamos la derivada por un valor $\Large \alpha$ (llamado ratio de aprendizaje) y lo restaremos a ls pesos correspondientes:

    $$\Large W = W - \alpha * \frac{\partial J(W, b)}{\partial W}$$

    $$\Large b = b - \alpha * \frac{\partial J(W, b)}{\partial b}$$

    6. Con los pesos actualizados, se vuelven a repetir los pasos anteriores hasta que el error, la diferencia entre las predicciones de la Red y los valores reales, sea aceptablemente pequeño.


## El Código

En la práctica, no se realiza la optimización de la Red Neuronal de a un ejemplo por vez. Sino que se aprovechan librerías de codigo que permiten utilizan las capacidades de las CPU, GPU y TPU modernas para realizar cálculos en forma simultáne sobre muchos ejemplos. 

En nuestro caso, emplearemos el lenguaje Python y principalmente la librería NumPy para realizar los cálculos sobre todos los ejemplos simultáneamente y obtener una mejor performance.

### El Modelo

Definiremos el modelo de nuestra Red Neuronal.




    






