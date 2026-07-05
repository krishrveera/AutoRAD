import torch

class Parameter:
    """A single bounded, optimizable scalar parameter.

    Wraps a named value together with its inclusive lower/upper bounds and
    enforces those bounds whenever a new candidate is assigned. Values are
    exposed as ``torch.double`` tensors for BoTorch interoperability.
    """

    def __init__(self, name, default, lower, higher):
        """Create a parameter.

        Parameters
        ----------
        name : str
            Parameter name (must match the ``RacingApp`` kwarg it maps to).
        default : float
            Initial / baseline value.
        lower, higher : float
            Inclusive lower and upper bounds for the search space.
        """
        self.name = name
        self._current = default
        self.lower = lower
        self.higher = higher

    @property
    def candidate(self):
        """Current value as a 0-D ``torch.double`` tensor."""
        return torch.tensor(self._current, dtype=torch.double)

    @candidate.setter
    def candidate(self, value):
        """Set the current value, validating it against the bounds.

        Accepts a Python float or a 0-D/1-D ``torch.Tensor`` (as returned by
        BoTorch).

        Raises
        ------
        ValueError
            If ``value`` falls outside ``[lower, higher]``.
        """
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
        """Bounds as a length-2 ``torch.double`` tensor ``[lower, higher]``."""
        return torch.tensor([self.lower, self.higher], dtype=torch.double)
    
class ParameterSpace:
    """An ordered collection of :class:`Parameter` objects.

    Provides the glue between BoTorch and the simulator: it exposes the
    stacked ``(2, d)`` bounds tensor BoTorch needs and converts candidate
    tensors back into the named ``kwargs`` dict consumed by ``RacingApp``.
    """

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
        """Human-readable, multi-line listing of every parameter and its bounds."""
        return "\n".join([f"{name}: {param._current} (bounds: [{param.lower}, {param.higher}])"
                          for name, param in self.params.items()])