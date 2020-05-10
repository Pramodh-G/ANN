
# Theory Questions for Assignment one

## Q1.Explain backward and forward propagation :

Forward and Backward prop are easily the most important elements in any neural network

Forward propagation is the process through which inputs pass successively from one layer to another till the out put has been reached.

Backward propagation is a process through which all the wieghts and biases get updated so as to minimize loss.

## Q2.Vectorization :

Every input in the initial layer is processed as a matrix of size $(1,4)$.

$$x = [x_1\:x_2\:x_3\:x_4]$$

The weight matrix for every neuron in the hidden layer would look like this : 

$$W^1_i = [w^1_1\:w^1_2\:w^1_3\:w^1_4]^T$$

where $W^j_i$ denotes weights of $i^{th}$neuron in $j^{th}$ layer.

the neuron returns

 $$a^1_i=\sigma(x.W^1_i)\:\forall i\in\{0,1,2,3\}$$

these $a^1_i$ 's are sent into the neuron in output layer which works in the same way.

BackProp :

the gradient is calculated of the loss wrt weights (for a single input).

the Loss function is :

$$L(\hat y,y) = {1 \over 2}(\hat y - y)^2$$

the gradient so calculated is a column vector of size $(4,1)$,and each element $(1,i)$ in it is given by :

$${\partial L \over \partial w^1_i}={\partial L \over \partial y}.{\partial y \over \partial w^2_i}.{\partial L \over \partial a^1_i}.{\partial a^1_i \over \partial w^1_i}$$

and the weights are then updated as :

$$W^1:=W^1-\alpha{\nabla_WL}$$

## Q3. List activation functions and their derivatives

- #### Sigmoid : 

$$\sigma(x) = {1 \over {1+e^x}}$$
$$\sigma^\prime(x)=\sigma(x)(1-\sigma(x))$$

- #### ReLU  :

 

$$relu(x) =
\left\{
    \begin{array}{ll}
        {x} & {if } x = 0 \\
        {0} & \: otherwise
    \end{array}
\right.$$

$$relu^\prime(x) =
\left\{
    \begin{array}{ll}
        {1} & {if } x \geq 0 \\
        {0} & otherwise
    \end{array}
\right.$$

- #### Leaky ReLU : 

$$lrelu(x) =
\left\{
    \begin{array}{ll}
        {x}  & {if } x \geq 0 \\
        {0.01x} & \: otherwise
    \end{array}
\right.$$


$$lrelu^\prime(x) =
\left\{
    \begin{array}{ll}
        {1} & {if } x \geq 0 \\
        {0.01} & {if } otherwise
    \end{array}
\right.$$

- #### tanh :

 $$\tanh(x) = {e^x - e^{-x} \over e^x + e^{-x}}$$

$$\tanh^\prime(x) = 1-\tanh^2(x)$$

- #### softmax for $k$ classes  : 

$$p_i = \frac{e^{a_i}}{\sum_{k=1}^N e^a_k}$$

with its derivative as :


$$
\frac{\partial p_i}{\partial a_j} = 
\begin{cases}p_i(1-p_j) & if \:i=j \cr
-p_j.p_i & if \: i \neq j 
\end{cases} $$ 