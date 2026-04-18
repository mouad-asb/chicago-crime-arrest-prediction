{{ config(materialized='table') }}

with base as (
    select * from {{ ref('stg_crimes') }}
),

-- historical arrest rate by crime type
crime_type_stats as (
    select
        crime_type,
        count(*)                                    as crime_type_total,
        sum(arrest::int)                            as crime_type_arrests,
        avg(arrest::int)                            as crime_type_arrest_rate
    from base
    group by crime_type
),

-- historical arrest rate by district
district_stats as (
    select
        district,
        count(*)                                    as district_total,
        sum(arrest::int)                            as district_arrests,
        avg(arrest::int)                            as district_arrest_rate
    from base
    group by district
),

-- historical arrest rate by hour of day
hour_stats as (
    select
        crime_hour,
        avg(arrest::int)                            as hour_arrest_rate
    from base
    group by crime_hour
),

-- historical arrest rate by day of week
dow_stats as (
    select
        crime_dow,
        avg(arrest::int)                            as dow_arrest_rate
    from base
    group by crime_dow
),

final as (
    select
        -- identifiers
        b.crime_id,
        b.case_number,

        -- time features
        b.crime_year,
        b.crime_month,
        b.crime_dow,
        b.crime_hour,
        -- is it a weekend?
        case when b.crime_dow in (0, 6) then 1 else 0 end  as is_weekend,
        -- time of day buckets
        case
            when b.crime_hour between 6 and 11  then 'morning'
            when b.crime_hour between 12 and 17 then 'afternoon'
            when b.crime_hour between 18 and 21 then 'evening'
            else 'night'
        end                                                 as time_of_day,

        -- crime features
        b.crime_type,
        b.is_domestic,
        b.location_description,
        b.fbi_code,
        b.beat,
        b.district,
        b.ward,
        b.community_area,

        -- aggregated historical features
        ct.crime_type_arrest_rate,
        ct.crime_type_total,
        ds.district_arrest_rate,
        ds.district_total,
        hs.hour_arrest_rate,
        dw.dow_arrest_rate,

        -- target
        b.arrest

    from base b
    left join crime_type_stats ct  on b.crime_type = ct.crime_type
    left join district_stats ds    on b.district = ds.district
    left join hour_stats hs        on b.crime_hour = hs.crime_hour
    left join dow_stats dw         on b.crime_dow = dw.crime_dow
)

select * from final