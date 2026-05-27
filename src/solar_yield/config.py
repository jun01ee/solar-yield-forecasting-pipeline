"""Configuration objects used by training and inference workflows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteConfig:
    """Physical and operational configuration for one solar asset."""

    latitude: float = -31.95
    longitude: float = 115.86
    timezone: str = "Australia/Perth"
    surface_tilt: float = 25.0
    surface_azimuth: float = 0.0
    albedo: float = 0.2

    def validate(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be between -90 and 90 degrees")
        if not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be between -180 and 180 degrees")
        if not 0 <= self.surface_tilt <= 90:
            raise ValueError("surface_tilt must be between 0 and 90 degrees")
        if not 0 <= self.surface_azimuth <= 360:
            raise ValueError("surface_azimuth must be between 0 and 360 degrees")
        if not 0 <= self.albedo <= 1:
            raise ValueError("albedo must be between 0 and 1")
