class  Node:
    def __init__(self, value):
        self.value = value
        self.probability = 0.0

    def set_probability(self, probability):
        """
        Set the probability for this node.

        Args:
            probability (float): Probability value to set
        """
        self.probability = probability
    def get_probability(self):
        """ Get the probability of this node.
        Returns:
            float: Probability value of this node
        """
        return self.probability
    def __repr__(self):
        """String representation of the node."""
        return f"Node(value={self.value}, probability={self.probability})"