with source as (
    select * from crimes_raw
),

cleaned as (
    select
        id                                             as crime_id,
        case_number,
        -- parse date into proper timestamp
        to_timestamp(date, 'MM/DD/YYYY HH12:MI:SS AM') as crime_timestamp,
        -- extract time components
        extract(year from to_timestamp(date, 'MM/DD/YYYY HH12:MI:SS AM'))::int as crime_year,
        extract(month from to_timestamp(date, 'MM/DD/YYYY HH12:MI:SS AM'))::int as crime_month,
        extract(dow from to_timestamp(date, 'MM/DD/YYYY HH12:MI:SS AM'))::int as crime_dow,
        extract(hour from to_timestamp(date, 'MM/DD/YYYY HH12:MI:SS AM'))::int as crime_hour,
        -- clean text fields
        trim(upper(primary_type))                   as crime_type,
        trim(upper(description))                    as crime_description,
        trim(upper(location_description))           as location_description,
        -- target variable
        arrest::boolean                             as arrest,
        domestic::boolean                           as is_domestic,
        -- location
        beat::int                                   as beat,
        district::int                               as district,
        ward::int                                   as ward,
        community_area::int                         as community_area,
        latitude,
        longitude,
        fbi_code,
        year

    from source
    where id is not null
      and date is not null
      and primary_type is not null
)

select * from cleaned