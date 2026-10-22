## Generate the AND + NOT Function Using a McCulloch–Pitts Neural Network

```python
# Q. Generate the AND + NOT function using a McCulloch–Pitts neural network

num_ip = int(input("ENTER NUMBER OF INPUTS="))
x1, x2 = [], []
for j in range(num_ip):
  x1.append(int(input("X1 = ")))
  x2.append(int(input("X2 = ")))

print("X1 = ",x1)
print("X2 = ",x2)

#AND
w1,w2 = 1,1
print("AND = ",[x1[i]*w1 + x2[i]*w2 for i in range(num_ip)])

#AND-NOT
w1,w2 = 1,-1
print("AND,NOT = ",[x1[i]*w1 + x2[i]*w2 for i in range(num_ip)])

#NOT
w1 = -1
print("NOT = ", [x1[i]*w1 for i in range(num_ip)])
```

## Generate the XOR Function Using a McCulloch–Pitts Neural Network

```python
# Q. Generate the XOR function using a McCulloch–Pitts neural network.

from sklearn.neural_network import MLPClassifier
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

clf = MLPClassifier(hidden_layer_sizes=(2,),
                    solver='lbfgs',
                    activation='logistic',
                    random_state=4)
clf.fit(X,y)
print("CORRECTED XOR FUNCTION IS=", clf.predict(X))
```


## Implement Hebb’s Rule

```python
# Q. Write a program to implement Hebb’s rule.

import numpy as np

x1 = x2 = np.array([1,1,1,-1,1,-1,1,1,1])
y = [1,-1]

w = np.zeros(9, dtype=int)
b = 0

print("\nOUTPUT WITH FIRST TARGET = 1")
w += x1 * y[0]
b += y[0]

print("\nOUTPUT WITH SECOND TARGET = -1")
w += x2 * y[1]
b += y[1]

print("NEW WEIGHT=",w)
print("NEW BIAS=",b)
```

## Implement the Error Backpropagation Algorithm

```python
# Q. Write a program to implement the Error Backpropagation Algorithm. And BackPropagation

import math
x, t = 1,1

c = int(input("Enter Learning Coefficient = "))
w1 = float(input("ENTER W1="))
w2 = float(input("ENTER W2="))
b1 = float(input("ENTER B1="))
b2 = float(input("ENTER B2="))

#fp
a1 = math.tanh(w1 * x + b1)
a2 = math.tanh(w2 * a1 + b2)

#bp
s2 = -2 * (1 - a2**2) * (t - a2)
s1 = (1 - a1**2) * w2 * s2

#Gradient Decent
w1 = w1 - c * s1 * x
w2 = w2 - c * s2 * a1
b1 = b1 - c * s1
b2 = b2 - c * s2

print("NEW W1 = ",w1)
print("NEW W2 = ",w2)
print("NEW B1 = ",b1)
print("NEW B2 = ",b2)
```

## Implement a Hopfield Network

```python
# Q. Write a program to implement a Hopfield Network.

import numpy as np

patterns = np.array([[1,1,-1,-1],[-1,-1,1,1]])

W = sum(np.outer(p,p) for p in patterns)
np.fill_diagonal(W,0)

test = np.array([1,1,-1,1])
output = np.sign(W @ test)
output[output==0] = 1

print("RETRIEVED/ CORRECTED OUTPUT IS = ",output)
```

## Implement a Kohonen Self-Organizing Map (SOM)

```python
# Q. Implement a Kohonen Self-Organizing Map (SOM).

!pip install minisom
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

colors = np.array([
    [0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],
])

color_name = np.array ([
    'red','blue','darkblue','skyblue','white','black','yellow'
])

som = MiniSom(20,30,3, sigma=1.0, learning_rate=0.5)
som.train_random(colors,400)

plt.imshow(som.get_weights())
plt.title("COLOR KOHONEN SOM")
plt.show()
```

## Demonstrate Linear Separation

```python
# Q. Write a program to demonstrate linear separation

import matplotlib.pyplot as plt
import numpy as np

# 1. Define two points to separate (e.g., two different classes)
p1, p2 = (1, 4), (3.5, 1.8)

# 2. Setup plot axes and labels
plt.xlim(-1, 6)
plt.ylim(-1, 8)
plt.xlabel("Sweetness")
plt.ylabel("Sourness")
plt.plot(p1[0], p1[1], "ro", markersize=10, label="Class A")
plt.plot(p2[0], p2[1], "bo", markersize=10, label="Class B")

# 3. Test different lines (slopes) to find a separating boundary
X = np.linspace(-0.5, 5, 100)
for angle in np.linspace(0.2, 1.5, 15):
    slope = np.tan(angle)

    # Check if the two points lie on opposite sides of the line y = slope * x
    side1 = (p1[1] - slope * p1[0]) > 0
    side2 = (p2[1] - slope * p2[0]) > 0

    # Green if successfully separated (opposite sides), Red if not
    color = "g-" if side1 != side2 else "r-"
    plt.plot(X, slope * X, color, alpha=0.6)

plt.title("Linear Separation Demonstration")
plt.legend()
plt.show()
```

## Use Fuzzy Logic to Find Ratios

```python
# Q. Use fuzzy logic to find ratios.

!pip install fuzzywuzzy
from fuzzywuzzy import fuzz, process

s1, s2 = "I love fuzzysforfuzzys", "I am loving fuzzysforfuzzys"

print("Fuzzywuzzy Ratio:", fuzz.ratio(s1, s2))
print("Fuzzywuzzy Partial Ratio:", fuzz.partial_ratio(s1, s2))
print("Fuzzywuzzy TokenSort Ratio:", fuzz.token_sort_ratio(s1, s2))
print("Fuzzywuzzy TokenSet Ratio:", fuzz.token_set_ratio(s1, s2))
print("Fuzzywuzzy W Ratio:", fuzz.WRatio(s1, s2), "\n")

query = "fuzzy for fuzzys"
choices = ["fuzzy for fuzzys", "fuzzy fuzzy", "g.for fuzzys"]

print("List of ratios:")
print(process.extract(query, choices), "\n")
print("Best among the above list:", process.extractOne(query, choices))
```