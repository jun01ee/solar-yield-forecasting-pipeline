"""Solar geometry and plane-of-array irradiance helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solar_yield.config import SiteConfig


def project_to_plane_of_array(predictions: pd.DataFrame, config: SiteConfig) -> pd.DataFrame:
    """Convert direct/diffuse attenuation factors into tilted-plane GTI components.

    pvlib is imported inside the function so the lightweight feature tests can run without
    requiring the full Databricks notebook dependency set.
    """

    import pvlib
    from pvlib.location import Location

    config.validate()
    result = predictions.copy()
    timestamps = pd.to_datetime(result["timestamp"])
    if timestamps.dt.tz is None:
        localized_index = timestamps.dt.tz_localize(config.timezone)
    else:
        localized_index = timestamps.dt.tz_convert(config.timezone)

    site = Location(config.latitude, config.longitude, tz=config.timezone)
    solar_position = site.get_solarposition(localized_index)
    clear_sky = site.get_clearsky(localized_index)

    zenith = solar_position["zenith"]
    attenuated_dni = clear_sky["dni"].to_numpy() * result["final_pred_direct"].to_numpy()
    attenuated_dhi = clear_sky["dhi"].to_numpy() * result["final_pred_diffuse"].to_numpy()
    attenuated_ghi = (attenuated_dni * np.cos(np.radians(zenith.to_numpy()))) + attenuated_dhi

    total_gti = pvlib.irradiance.get_total_irradiance(
        surface_tilt=config.surface_tilt,
        surface_azimuth=config.surface_azimuth,
        solar_zenith=zenith,
        solar_azimuth=solar_position["azimuth"],
        dni=attenuated_dni,
        ghi=attenuated_ghi,
        dhi=attenuated_dhi,
        albedo=config.albedo,
        model="isotropic",
    )

    result["gti_total"] = total_gti["poa_global"].to_numpy()
    result["gti_direct"] = total_gti["poa_direct"].to_numpy()
    result["gti_diffuse"] = total_gti["poa_diffuse"].to_numpy()
    return result
