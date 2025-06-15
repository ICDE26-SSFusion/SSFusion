from grace_lib import Compressor


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, name):
        return [tensor], None

    def decompress(self, tensors, ctx,name):
        tensor, = tensors
        return tensor

    def decompress_add(self, tensors, ctx,name):
        tensor, = tensors
        return tensor
