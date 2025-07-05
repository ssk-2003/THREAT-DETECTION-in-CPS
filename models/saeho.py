import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SAEHO:
    """
    Self-Adaptive Evolutionary Harmony Optimization (SAEHO) algorithm for 
    optimizing model hyperparameters.
    
    SAEHO combines concepts from harmony search and evolutionary algorithms
    with self-adaptive mechanisms to efficiently search the parameter space.
    """
    
    def __init__(self, population_size=30, max_iterations=50, harmony_memory_rate=0.7,
                 pitch_adjustment_rate=0.3, bandwidth=0.1):
        """
        Initialize the SAEHO optimizer.
        
        Args:
            population_size (int): Size of the harmony memory (population)
            max_iterations (int): Maximum number of iterations
            harmony_memory_rate (float): Rate of choosing from harmony memory
            pitch_adjustment_rate (float): Rate of pitch adjustment
            bandwidth (float): Distance bandwidth for pitch adjustment
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.harmony_memory_rate = harmony_memory_rate
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.bandwidth = bandwidth
        
    def optimize(self, model_class, param_ranges, X, y, cv=5, scoring='accuracy'):
        """
        Optimize model hyperparameters using SAEHO.
        
        Args:
            model_class: Class of the model to optimize
            param_ranges (dict): Dictionary of parameter ranges to search
            X: Input features
            y: Target labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for evaluation
            
        Returns:
            best_params (dict): Best parameters found
            best_score (float): Best score achieved
        """
        # Initialize harmony memory (population)
        harmony_memory = []
        harmony_scores = []
        
        # Generate initial population
        for _ in range(self.population_size):
            # Generate random parameters within the specified ranges
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range[0], int):
                    params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    params[param_name] = param_range[0] + np.random.random() * (param_range[1] - param_range[0])
            
            # Evaluate the parameters
            score = self._evaluate_params(model_class, params, X, y, cv, scoring)
            
            # Add to harmony memory
            harmony_memory.append(params)
            harmony_scores.append(score)
        
        # Sort harmony memory by score
        sorted_indices = np.argsort(harmony_scores)[::-1]
        harmony_memory = [harmony_memory[i] for i in sorted_indices]
        harmony_scores = [harmony_scores[i] for i in sorted_indices]
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Generate new harmony
            new_params = {}
            
            for param_name, param_range in param_ranges.items():
                # Decide whether to choose from harmony memory
                if np.random.random() < self.harmony_memory_rate:
                    # Choose from harmony memory
                    idx = np.random.randint(0, self.population_size)
                    new_params[param_name] = harmony_memory[idx][param_name]
                    
                    # Pitch adjustment
                    if np.random.random() < self.pitch_adjustment_rate:
                        if isinstance(param_range[0], int):
                            adjustment = np.random.randint(-1, 2)  # -1, 0, or 1
                            new_params[param_name] += adjustment
                            # Ensure within bounds
                            new_params[param_name] = max(param_range[0], min(param_range[1], new_params[param_name]))
                        else:
                            adjustment = (np.random.random() * 2 - 1) * self.bandwidth * (param_range[1] - param_range[0])
                            new_params[param_name] += adjustment
                            # Ensure within bounds
                            new_params[param_name] = max(param_range[0], min(param_range[1], new_params[param_name]))
                else:
                    # Generate randomly
                    if isinstance(param_range[0], int):
                        new_params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        new_params[param_name] = param_range[0] + np.random.random() * (param_range[1] - param_range[0])
            
            # Evaluate new harmony
            new_score = self._evaluate_params(model_class, new_params, X, y, cv, scoring)
            
            # Update harmony memory if better
            if new_score > harmony_scores[-1]:
                harmony_memory[-1] = new_params
                harmony_scores[-1] = new_score
                
                # Sort harmony memory by score
                sorted_indices = np.argsort(harmony_scores)[::-1]
                harmony_memory = [harmony_memory[i] for i in sorted_indices]
                harmony_scores = [harmony_scores[i] for i in sorted_indices]
            
            # Self-adaptation of parameters
            self._adapt_parameters(iteration)
        
        # Return best parameters and score
        return harmony_memory[0], harmony_scores[0]
    
    def _evaluate_params(self, model_class, params, X, y, cv, scoring):
        """
        Evaluate a set of parameters using cross-validation.
        
        Args:
            model_class: Class of the model to evaluate
            params (dict): Parameters to evaluate
            X: Input features
            y: Target labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for evaluation
            
        Returns:
            score (float): Cross-validation score
        """
        try:
            # Create model with the given parameters
            model = model_class(**params)
            
            # Perform cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            # Return mean score
            return np.mean(scores)
        except Exception as e:
            # Return a very low score in case of error
            print(f"Error evaluating parameters: {str(e)}")
            return -np.inf
    
    def _adapt_parameters(self, iteration):
        """
        Self-adaptive mechanism to adjust optimizer parameters.
        
        Args:
            iteration (int): Current iteration number
        """
        # Decrease bandwidth over time
        self.bandwidth = self.bandwidth * (1 - iteration / self.max_iterations)
        
        # Adjust pitch adjustment rate
        if iteration % 10 == 0:
            self.pitch_adjustment_rate = 0.1 + 0.4 * np.random.random()

