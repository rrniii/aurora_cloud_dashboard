"""Shared instrument identity and operational presentation metadata.

This module deliberately contains no Panel, xarray, or filesystem access.  It
is the stable contract used by the browser dashboard and mobile API when they
refer to an instrument, its icon, or its assigned PDU outlet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentContract:
    id: str
    title: str
    system_image: str
    browser_key: str
    quicklook_subdir: str
    science_prefixes: tuple[str, ...]
    housekeeping_prefixes: tuple[str, ...] = ()
    visible: bool = True
    summary_supported: bool = True
    pdu_outlet: int | None = None
    pdu_title: str | None = None
    automatic_phase_labels: tuple[tuple[str, str], ...] = ()
    legacy_science_latest_alias: bool = False


INSTRUMENTS: tuple[InstrumentContract, ...] = (
    InstrumentContract(
        "power",
        "Aurora Power Supply",
        "battery.100percent",
        "power",
        "power",
        ("power__summary",),
        ("power__HK_APS", "power__hk_aps"),
    ),
    InstrumentContract(
        "ceilometer",
        "Ceilometer",
        "laser.burst",
        "Ceilometer",
        "ceilometer",
        ("ceilometer",),
        ("ceilometer__HK_Ceilometer", "ceilometer__hk_ceilometer"),
        pdu_outlet=5,
        pdu_title="CL61",
        automatic_phase_labels=(("fan_high", "On with Heater/Blower"),),
        legacy_science_latest_alias=True,
    ),
    InstrumentContract(
        "cloud-radar",
        "Cloud Radar",
        "dot.radiowaves.left.and.right",
        "Cloud Radar",
        "cloud_radar",
        ("cloud_radar",),
        ("cloud_radar__HK_Radar", "cloud_radar__hk_radar"),
        pdu_outlet=6,
        pdu_title="Cloud Radar",
        legacy_science_latest_alias=True,
    ),
    InstrumentContract(
        "hatpro",
        "Scanning Microwave Radiometer",
        "antenna.radiowaves.left.and.right",
        "Scanning Microwave Radiometer",
        "hatpro",
        ("hatpro",),
        pdu_outlet=8,
        pdu_title="HATPRO",
        legacy_science_latest_alias=True,
    ),
    InstrumentContract(
        "vaisalamet",
        "Meteorology",
        "cloud.sun",
        "vaisalamet",
        "vaisalamet",
        ("vaisalamet__summary",),
        ("vaisalamet__HK_Met", "vaisalamet__hk_met"),
    ),
    InstrumentContract(
        "asfs-logger",
        "Radiation",
        "sun.max",
        "asfs-logger",
        "asfs_logger",
        ("asfs_logger__summary",),
        ("asfs_logger__HK_ASFS", "asfs_logger__hk_asfs"),
    ),
    InstrumentContract(
        "ops-monitor",
        "Operations",
        "gauge.with.dots.needle.bottom.50percent",
        "ops-monitor",
        "ops_monitor",
        ("ops_monitor__summary",),
        ("ops_monitor__HK_Operations", "ops_monitor__hk_operations"),
    ),
    InstrumentContract(
        "wxcam",
        "WXcam",
        "video",
        "wxcam",
        "wxcam",
        ("wxcam",),
        ("wxcam__HK_WXcam", "wxcam__hk_wxcam"),
        summary_supported=False,
        legacy_science_latest_alias=True,
    ),
    InstrumentContract(
        "uas",
        "UAS",
        "airplane",
        "uas",
        "uas",
        ("uas__summary",),
        visible=False,
        summary_supported=False,
        pdu_outlet=4,
        pdu_title="UAS",
    ),
)

INSTRUMENT_BY_ID = {instrument.id: instrument for instrument in INSTRUMENTS}
INSTRUMENT_BY_BROWSER_KEY = {instrument.browser_key: instrument for instrument in INSTRUMENTS}

PDU_INSTRUMENTS = tuple(
    sorted(
        (
            instrument
            for instrument in INSTRUMENTS
            if instrument.pdu_outlet is not None
        ),
        key=lambda instrument: instrument.pdu_outlet or 0,
    )
)

SCIENCE_DC_INSTRUMENTS = tuple(
    instrument
    for instrument in INSTRUMENTS
    if instrument.id in {"vaisalamet", "asfs-logger"}
)


def browser_options(*, housekeeping: bool = False) -> dict[str, str]:
    """Return the existing browser labels and internal instrument keys."""
    order = (
        "ceilometer",
        "cloud-radar",
        "vaisalamet",
        "asfs-logger",
        "power",
        *(("ops-monitor",) if housekeeping else ()),
        "wxcam",
        "hatpro",
    )
    return {
        INSTRUMENT_BY_ID[instrument_id].title: INSTRUMENT_BY_ID[instrument_id].browser_key
        for instrument_id in order
    }


def science_options() -> dict[str, str]:
    """Return Science Quicklook choices without widening the data browser.

    UAS publishes daily image products but has no generic Zarr-backed browser
    contract.  Keeping this list separate prevents it from leaking into either
    the Interactive Data Browser or House Keeping Quicklooks.
    """
    options = browser_options()
    options[INSTRUMENT_BY_ID["uas"].title] = INSTRUMENT_BY_ID["uas"].browser_key
    return options
