
DATAPACKAGE: HOUSEHOLD DATA
===========================================================================

by Open Power System Data: http://www.open-power-system-data.org/

Package Version: 2020-04-15

Detailed household load and solar generation in minutely to hourly
resolution

This data package contains measured time series data for several small
businesses and residential households relevant for household- or
low-voltage-level power system modeling. The data includes solar power
generation as well as electricity consumption (load) in a resolution up to
single device consumption. The starting point for the time series, as well
as data quality, varies between households, with gaps spanning from a few
minutes to entire days. All measurement devices provided cumulative energy
consumption/generation over time. Hence overall energy
consumption/generation is retained, in case of data gaps due to
communication problems. Measurements were conducted 1-minute intervals,
with all data made available in an interpolated, uniform and regular time
interval. All data gaps are either interpolated linearly, or filled with
data of prior days. Additionally, data in 15 and 60-minute resolution is
provided for compatibility with other time series data. Data processing is
conducted in Jupyter Notebooks/Python/pandas.

The data package covers the geographical region of 11 households in southern Germany.

We follow the Data Package standard by the Frictionless Data project, a
part of the Open Knowledge Foundation: http://frictionlessdata.io/


Documentation and script
===========================================================================

This README only contains the most basic information about the data package.
For the full documentation, please see the notebook script that was used to
generate the data package. You can find it at:

https://nbviewer.jupyter.org/github/isc-konstanz/household_data/blob/2020-04-15/main.ipynb

Or on GitHub at:

https://github.com/isc-konstanz/household_data/blob/2020-04-15/main.ipynb

License and attribution
===========================================================================

Data license: 
Creative Commons Attribution-International

Script license:
    [MIT License](https://opensource.org/licenses/MIT)

Attribution:
    Attribution in Chicago author-date style should be given as follows:
    "Open Power System Data. 2020. Data Package Household Data. Version
    2020-04-15.
    https://data.open-power-system-data.org/household_data/2020-04-15/.
    (Primary data from various sources, for a complete list see URL)."


Version history
===========================================================================

* 2020-04-15 Second release of all CoSSMic households
* 2017-11-10 Initial release of all CoSSMic households
* 2017-09-01 Improve adjustments function and fix redundant rounding
* 2017-08-01 Initial upload


Resources
===========================================================================

* [Package description page](http://data.open-power-system-data.org/household_data/2020-04-15/)
* [ZIP Package](http://data.open-power-system-data.org/household_data/opsd-household_data-2020-04-15.zip)
* [Script and documentation](https://github.com/isc-konstanz/household_data/blob/2020-04-15/main.ipynb)
* [Original input data](http://data.open-power-system-data.org/household_data/2020-04-15/original_data/)


Sources
===========================================================================

* [CoSSMic](http://isc-konstanz.de/isc/institut/oeffentliche-projekte/abgeschl-projekte/eu/cossmic.html)


Field documentation
===========================================================================


household_data_1min_singleindex.csv
---------------------------------------------------------------------------

* utc_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%SZ
    - Description: Start of timeperiod in Coordinated Universal Time
* cet_cest_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%S%z
    - Description: Start of timeperiod in Central European (Summer-) Time
* interpolated
    - Type: string
    - Description: marker to indicate which columns are missing data in source data and has been interpolated (e.g. DE_KN_Residential1_grid_import;)
* DE_KN_industrial1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_1
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_2
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_charge
    - Type: number (float)
    - Description: Battery charging energy in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_decharge
    - Type: number (float)
    - Description: Energy in kWh
* DE_KN_industrial3_area_offices
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_1
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_2
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_3
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_4
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_compressor
    - Type: number (float)
    - Description: Compressor energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_aggregate
    - Type: number (float)
    - Description: Cooling aggregate energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_pumps
    - Type: number (float)
    - Description: Cooling pumps energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_1
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_2
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_3
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_4
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_5
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_facade
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_roof
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ventilation
    - Type: number (float)
    - Description: Ventilation energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_public1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_public2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_residential1_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential1_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the suburban area in kWh
* DE_KN_residential1_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential2_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential3_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a residential building, located in the urban area in kWh
* DE_KN_residential4_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential4_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential5_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential6_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential6_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh


household_data_15min_singleindex.csv
---------------------------------------------------------------------------

* utc_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%SZ
    - Description: Start of timeperiod in Coordinated Universal Time
* cet_cest_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%S%z
    - Description: Start of timeperiod in Central European (Summer-) Time
* interpolated
    - Type: string
    - Description: marker to indicate which columns are missing data in source data and has been interpolated (e.g. DE_KN_Residential1_grid_import;)
* DE_KN_industrial1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_1
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_2
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_charge
    - Type: number (float)
    - Description: Battery charging energy in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_decharge
    - Type: number (float)
    - Description: Energy in kWh
* DE_KN_industrial3_area_offices
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_1
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_2
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_3
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_4
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_compressor
    - Type: number (float)
    - Description: Compressor energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_aggregate
    - Type: number (float)
    - Description: Cooling aggregate energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_pumps
    - Type: number (float)
    - Description: Cooling pumps energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_1
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_2
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_3
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_4
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_5
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_facade
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_roof
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ventilation
    - Type: number (float)
    - Description: Ventilation energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_public1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_public2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_residential1_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential1_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the suburban area in kWh
* DE_KN_residential1_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential2_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential3_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a residential building, located in the urban area in kWh
* DE_KN_residential4_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential4_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential5_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential6_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential6_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh


household_data_60min_singleindex.csv
---------------------------------------------------------------------------

* utc_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%SZ
    - Description: Start of timeperiod in Coordinated Universal Time
* cet_cest_timestamp
    - Type: datetime
    - Format: fmt:%Y-%m-%dT%H%M%S%z
    - Description: Start of timeperiod in Central European (Summer-) Time
* interpolated
    - Type: string
    - Description: marker to indicate which columns are missing data in source data and has been interpolated (e.g. DE_KN_Residential1_grid_import;)
* DE_KN_industrial1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_1
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial1_pv_2
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial warehouse building in kWh
* DE_KN_industrial2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_charge
    - Type: number (float)
    - Description: Battery charging energy in a industrial building of a business in the crafts sector in kWh
* DE_KN_industrial2_storage_decharge
    - Type: number (float)
    - Description: Energy in kWh
* DE_KN_industrial3_area_offices
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_1
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_2
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_3
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_area_room_4
    - Type: number (float)
    - Description: Energy consumption of an area, consisting of several smaller loads, in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_compressor
    - Type: number (float)
    - Description: Compressor energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_aggregate
    - Type: number (float)
    - Description: Cooling aggregate energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_cooling_pumps
    - Type: number (float)
    - Description: Cooling pumps energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_1
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_2
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_3
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_4
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_machine_5
    - Type: number (float)
    - Description: Energy consumption of an industrial- or research-machine in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_facade
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_pv_roof
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_industrial3_ventilation
    - Type: number (float)
    - Description: Ventilation energy consumption in a industrial building, part of a research institute in kWh
* DE_KN_public1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_public2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a school building, located in the urban area in kWh
* DE_KN_residential1_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential1_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential1_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the suburban area in kWh
* DE_KN_residential1_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential2_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the suburban area in kWh
* DE_KN_residential2_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the suburban area in kWh
* DE_KN_residential3_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential3_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential3_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential3_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_ev
    - Type: number (float)
    - Description: Electric Vehicle charging energy in a residential building, located in the urban area in kWh
* DE_KN_residential4_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential4_heat_pump
    - Type: number (float)
    - Description: Heat pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential4_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential4_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential5_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_refrigerator
    - Type: number (float)
    - Description: Refrigerator energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential5_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential apartment, located in the urban area in kWh
* DE_KN_residential6_circulation_pump
    - Type: number (float)
    - Description: Circulation pump energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_dishwasher
    - Type: number (float)
    - Description: Dishwasher energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_freezer
    - Type: number (float)
    - Description: Freezer energy consumption in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_export
    - Type: number (float)
    - Description: Energy exported to the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_grid_import
    - Type: number (float)
    - Description: Energy imported from the public grid in a residential building, located in the urban area in kWh
* DE_KN_residential6_pv
    - Type: number (float)
    - Description: Total Photovoltaic energy generation in a residential building, located in the urban area in kWh
* DE_KN_residential6_washing_machine
    - Type: number (float)
    - Description: Washing machine energy consumption in a residential building, located in the urban area in kWh


Feedback
===========================================================================

Thank you for using data provided by Open Power System Data. If you have
any question or feedback, please do not hesitate to contact us.

For this data package, contact:
Adrian Minde <adrian.minde@isc-konstanz.de>

For general issues, find our team contact details on our website:
http://www.open-power-system-data.org