"""
Download ECMWF forecast data from the MARS service.

To use this script, store your MARS crendentials to a file
"mars-credentials.json":

{
    "key": "<your_access_key>",
    "email": "<your_email"
}


Examples:

Download 24 h from a single forecast:
    python download_ecmwf.py -s 2019-05-01T00 -d 24
or
    python download_ecmwf.py -s 2019-05-01 -d 24
Creates 'ecmwf_2019-05-01T00_24h.nc' file.

Download 12 h from sequental runs and store a single concatenated file:
    python download_ecmwf.py -s 2019-05-01T00 -e 2019-05-03T00
Creates 'ecmwf_2019-05-01_2019-05-03.nc' file. Last time stamp is 2019-05-03T23.

Download a whole month
    python download_ecmwf.py -m 2019-05
Creates 'ecmwf_2019-05.nc' file.

Requires ecmwf-api-client:
    pip install ecmwf-api-client

See: https://confluence.ecmwf.int/display/WEBAPI/Access+MARS
"""
from ecmwfapi import ECMWFService
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, HOURLY
from dateutil.parser import parse as dateparse
import datetime
import numpy
import netCDF4
import os
import json


def download_forecast(date, hours=11):
    """
    Downloads forecast data starting from the given time stamp.

    Files are stored to disk in 'ecmwf_YYYY-MM-DDTHH_XXh.nc' format.
    YYYY-MM-DDTHH - forecast start date
    XX            - number of hours

    :arg date: a datetime object, e.g. datetime(2019, 5, 1, 12)
    :kwarg int hours: number of hourly steps to download, e.g. 12
    """
    with open('mars-credentials.json', 'r') as f:
        d = json.load(f)
        key = d['key']
        email = d['email']

    server = ECMWFService('mars',
                          url='https://api.ecmwf.int/v1',
                          key='80b00604dbf257efd28aa352edb70e32',
                          email='tuomas.karna@fmi.fi')

    step_str = '/'.join(map(str, numpy.arange(int(hours + 1))))
    output_file = 'ecmwf_{:%Y-%m-%dT%H}_{:02d}h.nc'.format(date, hours)
    request = {
        'class': 'od',
        'date': date.strftime('%Y-%m-%d'),
        'expver': '73',
        'levtype': 'sfc',
        'param': '134.128/144.128/165.128/166.128/167.128/168.128/169.128/175.128/228.128',
        'step': step_str,
        'stream': 'oper',
        'area': '66/-5/48/31',
        'grid': '0.20689655172413793/0.1125',
        'format': 'netcdf',
        'time': date.strftime('%H:%M:%S'),
        'type': 'fc',
    }
    # download to tmp file
    tmpfile = 'tmp_' + output_file
    server.execute(request, tmpfile)
    # copy to convert all variables to float datatype
    concat_netcdf_files([tmpfile], output_file,
                        rm_source_files=True, verbose=False)
    # deaccumulate accumulated fields
    deaccumulate_fields(output_file, ['ssrd', 'strd', 'tp', 'sf'])
    # compute specific humidity
    compute_specific_humidity(output_file)
    return output_file


def concat_netcdf_files(file_list, output_file, rm_source_files=False,
                        verbose=True):
    """
    Concatenate netCDF files into one along the record dimension.

    :arg file_list: list of netcdf files to be merged (in correct order)
    :arg output_file: name of the output file
    :kwarg rm_source_files: if True, removes the source files when finished
    """
    if verbose:
        print('Merging files into {:}'.format(output_file))

    # open multi-file dataset and copy it to a output dataset
    with netCDF4.MFDataset(file_list) as src, netCDF4.Dataset(output_file, 'w') as dst:
        # copy global attributes all at once via dictionary
        for key in src.__dict__:
            if key[0] != '_':
                dst.setncattr(key, src.__dict__[key])
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        # copy all file data except for the excluded
        for name, variable in src.variables.items():
            dtype = numpy.float32
            if name == 'time':
                dtype == numpy.float64
            dst.createVariable(name, dtype, variable.dimensions)
            dst[name][:] = variable[:]
            # copy variable attributes all at once via dictionary
            for key in variable.__dict__:
                if key[0] != '_' and key not in ['scale_factor', 'add_offset', 'missing_value']:
                    try:
                        dst[name].setncattr(key, variable.__dict__[key])
                    except TypeError:
                        # silently omit attributes with unsupported data type
                        pass

    if rm_source_files:
        for f in file_list:
            if os.path.isfile(f):
                os.remove(f)


def deaccumulate_fields(ncfile, varname_list):
    """
    Deaccumulate fields that have been accumulated over time.

    Creates a new variable "deacc_X" where X is the original variable name.

    Precipitation:

    To obtain average precipitation per hour between 00:00 and 06:00,
    download Total Precipitation with time 00:00 and steps 0 (tp0) and step 6
    (tp6), then calculate:

    ( tp6 - tp0 ) / ( 6 - 0 )

    Radiation:

    This parameter is accumulated over a particular time period which depends
    on the data extracted. The units are joules per square metre (J m-2). To
    convert to watts per square metre (W m-2), the accumulated values should be
    divided by the accumulation period expressed in seconds.

    See:
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=56658233
    https://apps.ecmwf.int/codes/grib/param-db?id=169
    """
    with netCDF4.Dataset(ncfile, 'r+') as src:
        src.set_auto_scale(False)
        time = src['time'][:]
        assert src['time'].units.split(' ')[0] == 'hours'
        timestep = numpy.diff(time)
        assert numpy.allclose(timestep, timestep[0]*numpy.ones_like(timestep))
        timestep = timestep[0]
        for varname in varname_list:
            use_seconds = varname in ['ssrd', 'strd']
            scalar = timestep*3600. if use_seconds else timestep
            var = src[varname]
            newvarname = 'deacc_' + varname
            if newvarname not in src.variables:
                src.createVariable(newvarname, var.dtype, var.dimensions)
            newvar = src[newvarname]
            vals = var[:].astype(numpy.float32)
            newvals = numpy.zeros_like(vals)
            newvals[:-1, :, :] = numpy.diff(vals, axis=0)
            newvals /= scalar
            # remove negative values
            newvals[newvals < 0.0] = 0.0
            newvals[-1, :, :] = numpy.ma.masked
            newvar[:] = newvals
            # copy variable attributes
            for key in var.ncattrs():
                if key[0] != '_' and key not in ['scale_factor', 'add_offset', 'missing_value']:
                    newvar.setncattr(key, var.getncattr(key))
            units = {
                'ssrd': 'W m-2',
                'strd': 'W m-2',
            }
            if varname in units:
                newvar.setncattr('units', units[varname])


def specific_humidity(T, p):
    """
    Compute specific humidity from dew point temperature and surface pressure.

    Specific humidity is calculated over water and ice using equations 7.4 and
    7.5 from Part IV, Physical processes section (Chapter 7, section 7.2.1b) in
    the documentation of the IFS for CY41R2. Use the 2m dew point temperature
    and surface pressure (which is approximately equal to the pressure at 2m)
    in these equations. The constants in 7.4 are to be found in Chapter 12
    (of Part IV: Physical processes) and the parameters in 7.5 should be set
    for saturation over water because the dew point temperature is being used.

    See: https://confluence.ecmwf.int/display/CKB/ERA+datasets%3A+near-surface+humidity
    See: https://www.ecmwf.int/en/elibrary/16648-part-iv-physical-processes

    :arg T: dew point temperature in Kelvins (numpy array)
    :arg p: surface pressure in Pascals (numpy array)

    :returns: specific humidity in a numpy array
    """
    R_dry = 287.0597
    R_vap = 461.5250
    T0 = 273.16
    a1 = 611.21
    # a3 and a4 for over water
    a3 = 17.502
    a4 = 32.19
    # eq 7.5
    e_sat = a1 * numpy.exp(a3 * (T - T0)/(T - a4))
    # eq 7.4
    r = R_dry/R_vap
    q_sat = r * e_sat / (p - (1 - r)*e_sat)
    return q_sat


def compute_specific_humidity(ncfile, dewp_varname='d2m', spres_varname='sp',
                              shumi_varname='q2'):
    """
    Compute specific humidity from dewpoint and surface pressure fields.
    """
    with netCDF4.Dataset(ncfile, 'r+') as src:
        q2 = src.createVariable(shumi_varname, numpy.float32,
                                ('time', 'latitude', 'longitude', ))
        q2.units = 'g/g'
        q2.long_name = '2 m specific humidity'

        d2m = src.variables[dewp_varname]
        sp = src.variables[spres_varname]
        for i in range(d2m.shape[0]):
            q2[i] = specific_humidity(d2m[i], sp[i])


def download_range(starttime, endtime, hours=11,
                   concatenate=False,
                   concat_fmt=None):
    """
    Download data from multiple forecast runs.

    :arg datetime starttime: start of the first forecast run
    :arg datetime endtime: start of the last forecast run
    :kwarg int hours: number of hours to download from each forecast
    :kwarg concatenate: if True, will concatenate files into one
    :kwarg concat_fmt: formatting string for the concatenated file
    """
    file_list = []
    for date in rrule(HOURLY, dtstart=starttime, until=endtime, interval=12):
        print('\nFetching {:%Y-%m-%dT%H}'.format(date))
        f = download_forecast(date, hours=hours)
        file_list.append(f)

    if concatenate:
        if concat_fmt is None:
            concat_fmt = 'ecmwf_{start:%Y-%m-%dT%HH}_{end:%Y-%m-%d%HH}.nc'
        output_file = concat_fmt.format(start=starttime, end=endtime)
        concat_netcdf_files(file_list, output_file, rm_source_files=True)


if __name__ == '__main__':
    import argparse

    header = "Download ECMWF bi-daily forecast files from MARS server in netCDF format."
    parser = argparse.ArgumentParser(
        description=header,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-s', '--startdate',
                        help='Start date of first forecast run, in format "YYYY-MM-DDTHH" or "YYYY-MM-DD", e.g. "2006-05-01T00".')
    parser.add_argument('-e', '--enddate',
                        help='Start date of last forecast run, in format "YYYY-MM-DDTHH" or "YYYY-MM-DD", e.g. "2006-05-03T00" (optional). If omitted, will only download one forecast.')
    parser.add_argument('-m', '--month',
                        help='Download an entire month, expects format "YYYY-MM". Downloads the whole month from sequential 12 h forecasts.')
    parser.add_argument('-d', '--duration', type=int, default=11,
                        help='Forecast duration in hours.')
    parser.add_argument('-c', '--concatenate', action='store_true',
                        help='Concatenate downloaded files into a single file. File format is "ecmwf_YYYY-MM-DD_YYYY-MM-DD.nc" for an arbitrary date range, or "ecmwf_YYYY-MM.nc" for a whole month. Overrides duration to 12 h.')
    args = parser.parse_args()

    concat_fmt = None
    if args.startdate is None and args.enddate is None:
        # download entire month
        if args.month is None:
            parser.print_help()
            raise ValueError('ERROR: Either "-s" or "-m" flag must be provided')
        year, month = [int(v) for v in args.month.split('-')]
        s = datetime.datetime(year, month, 1)
        e = s + relativedelta(months=1, hours=-12)
        concat_fmt = 'ecmwf_{start:%Y-%m}.nc'
    else:
        s = dateparse(args.startdate)
        if args.enddate is None:
            e = s
        else:
            e = dateparse(args.enddate)
    hours = args.duration
    if args.concatenate:
        # can only concatenate datasets if duration is 11 h
        hours = 11
    assert hours <= 90, 'hours must be <= 90. ECMWF only stores first 90 h with 1 h resolution.'

    download_range(s, e, hours, concatenate=args.concatenate, concat_fmt=concat_fmt)
