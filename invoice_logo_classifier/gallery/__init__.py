"""Gallery module: manages embedding gallery for unseen vendor logos and nearest-neighbor retrieval."""

from invoice_logo_classifier.gallery.extractor import LogoEmbeddingExtractor
from invoice_logo_classifier.gallery.gallery_store import GalleryMatch, VendorGallery

__all__ = [
    "GalleryMatch",
    "LogoEmbeddingExtractor",
    "VendorGallery",
]
