import math


class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # allows us to add a numeric value to a Value object
        out = Value(data=self.data + other.data, _children=(self, other),  _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # allows us to add a numeric value to a Value object
        out = Value(data=self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(data=self.data**other, _children=(self, ), _op=f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad # local derivative is the power rule: n(x^(n - 1))
        out._backward = _backward

        return out
        
    
    def __rmul__(self, other): # other * self
        """
        This is a fallback (if Python can't execute 2 * (Value Object), Python knows to check this function for how to proceed with the multiplication)
        
        Allows us to define the case of 2 * (Value Object) s.t. 2.__mul__(Value Object) works as intended (__rmul__ will call (Value Object) * 2 - which is defined above - instead of 2 * (Value Object))
        """
        return self * other


    def __truediv__(self, other): # self / other
        """
        We can rewrite a / b as:
        a * (1/b)
        a * (b^(-1))
        """
        return self * other**-1


    def __neg__(self): # -self
        return self * -1


    def __sub__(self, other): # self - other
        return self + (-other)


    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(data=t, _children=(self, ), _op='tanh')

        def _backward():
            self.grad += 1 - t**2 * out.grad
        out._backward = _backward
        
        return out


    def exp(self):
        """
        Mirrors `tanh` in the sense that `exp` is a function that takes in a single scalar value and outputs a single scalar value
        """
        x = self.data
        out = Value(data=math.exp(x), _children=(self, ), _op='exp')

        def _backward():
            """
            d/dx of e^x is e^x

            And += to take care of multi-variate case
            """
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: # you can think of v._prev as what the root (or current) node point to - the resulting topo is a little backwards, but we are doing back prop after all!
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()