from grace_lib import Memory

class ResidualGlobalTopkMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0, afa=0):
        self.residuals = {}
        self.residuals_global = {}
        self.tensor_decompressed_={}
        self.beta = beta
        self.gamma = gamma
        self.afa = afa
        
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""

        # return tensor
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] +self.afa * self.residuals_global[name] + self.gamma * tensor
        return tensor


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx,name)

        residual = tensor - tensor_decompressed
        self.residuals[name] = residual
        self.tensor_decompressed_[name]=tensor_decompressed

        return tensor_decompressed

    def compensate_gtopk(self, tensors_aggregated_avg, name):
        """Update the tensor with the residuals."""
        if name in self.residuals_global:
            tensors_aggregated_avg = self.beta * self.residuals_global[name] + self.gamma * tensors_aggregated_avg
        return tensors_aggregated_avg


    def update_gtopk(self,ctx,tensor_decompressed, tensors_aggregated_avg, original_tensor, name):
        """Update the residuals."""

        residual = self.tensor_decompressed_[name] - tensors_aggregated_avg

        self.residuals_global[name] = residual
