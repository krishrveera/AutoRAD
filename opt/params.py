import torch

class Parameter:
    def __init__(self, name, default, lower, higher):
        self.name = name
        self._current = default
        self.lower = lower
        self.higher = higher

    @property
    def candidate(self):
        return torch.tensor(self._current, dtype=torch.double)
    
    @candidate.setter
    def candidate(self, value):
        # Safely handle if 'value' comes back from BoTorch as a 1D or 0D tensor
        if isinstance(value, torch.Tensor):
            val_to_check = value.item()
        else:
            val_to_check = float(value)

        # Validate against bounds
        if val_to_check < self.lower or val_to_check > self.higher:
            raise ValueError(
                f"Value {val_to_check:.6f} for '{self.name}' is out of bounds "
                f"[{self.lower}, {self.higher}]"
            )
        self._current = val_to_check
        
    @property
    def range(self):
        return torch.tensor([self.lower, self.higher], dtype=torch.double)
    
class ParameterSpace:
    def __init__(self, parameters):
        """
        Initializes the space with a list of Parameter objects.
        Python dictionaries preserve order, acting as our map.
        """
        self.params = {p.name: p for p in parameters}
        self.names = list(self.params.keys())
        
    @property
    def bounds(self):
        """
        Combines all individual bounds into the exact (2, d) tensor BoTorch needs.
        """
        # Collect all individual ranges: shape becomes a list of 1D tensors
        ranges = [p.range for p in self.params.values()]
        
        # Stack them side-by-side (dim=1) to get the [Lower Bounds, Upper Bounds] format
        # Returns shape: (2, number_of_parameters)
        return torch.stack(ranges, dim=1)
        
    def tensor_to_kwargs(self, candidate_tensor):
        """
        Takes a BoTorch candidate tensor, updates the internal parameters, 
        and unpacks them into a named dictionary (kwargs) for your program.
        """
        # Squeeze the tensor in case BoTorch passes shape (1, d) instead of (d,)
        flat_tensor = candidate_tensor.flatten()
        
        if len(flat_tensor) != len(self.names):
            raise ValueError(
                f"Tensor dimension {len(flat_tensor)} does not match "
                f"parameter count {len(self.names)}."
            )
            
        kwargs = {}
        for i, name in enumerate(self.names):
            # 1. Update the Parameter's internal state (triggers your boundary checks)
            self.params[name].candidate = flat_tensor[i]
            
            # 2. Add the safe float value to the kwargs dictionary
            # Using _current avoids returning a tensor object to your underlying program
            kwargs[name] = self.params[name]._current 
            
        return kwargs
    
    def __str__(self):
        return "\n".join([f"{name}: {param._current} (bounds: [{param.lower}, {param.higher}])" 
                          for name, param in self.params.items()])