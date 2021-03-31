head(flights)
tail(flights)
glimpse(flights)
select(flights,dep_time,arr_time,flight)
filter(flights,month==1,day==1,origin=="EWR")
filter(flights, carrier=="AA" | carrier=="UA")
flights %>%
  select(carrier, dep_delay) %>%
  filter(dep_delay > 60)
flights %>%
  select(carrier, dep_delay) %>%
  arrange(dep_delay)
flights <- flights %>% mutate(Speed = distance/air_time*60)
flights %>%
  group_by(dest) %>%
  summarise(avg_delay = mean(arr_delay, na.rm=TRUE))
flights %>%
  group_by(month, day) %>%
  summarise(flight_count = n()) %>%
  arrange(desc(flight_count))
