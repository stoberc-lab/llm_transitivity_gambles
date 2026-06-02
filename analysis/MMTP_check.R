library(dplyr)
library(tidyverse)

#MMTP Check
test_mmtp <- function(df) {
  triads <- list(
    list(cols = c("A_B", "B_C", "A_C"), name = "ABC"),
    list(cols = c("A_B", "B_D", "A_D"), name = "ABD"),
    list(cols = c("A_B", "B_E", "A_E"), name = "ABE"),
    list(cols = c("A_C", "C_D", "A_D"), name = "ACD"),
    list(cols = c("A_C", "C_E", "A_E"), name = "ACE"),
    list(cols = c("A_D", "D_E", "A_E"), name = "ADE"),
    list(cols = c("B_C", "C_D", "B_D"), name = "BCD"),
    list(cols = c("B_C", "C_E", "B_E"), name = "BCE"),
    list(cols = c("B_D", "D_E", "B_E"), name = "BDE"),
    list(cols = c("C_D", "D_E", "C_E"), name = "CDE")
  )
  
  checked_df <- df
  
  for (triad in triads) {
    columns <- triad$cols
    column_name <- triad$name
    
    p_ab <- checked_df[[columns[1]]]
    p_bc <- checked_df[[columns[2]]]
    p_ac <- checked_df[[columns[3]]]
    
    #  The intransitive condition for MMTP
    intransitive_condition <- (p_ab + p_bc - p_ac > 1) | 
      ((1 - p_ab) + (1 - p_bc) - (1 - p_ac) > 1)
    
    # Add a new column with the result (1 if intransitive, 0 otherwise)
    checked_df[[column_name]] <- as.integer(intransitive_condition)
  }
  
  # Get the names of all the new 'mmtp_...' columns we just created
  mmtp_cols <- sapply(triads, function(t) t$name)
  
  # If the sum of violations is 0, result is 1. Otherwise, result is 0.
  checked_df$MMTP_check_total <- ifelse(rowSums(checked_df[, mmtp_cols]) == 0, 1, 0) 
 
  return(checked_df)
}
