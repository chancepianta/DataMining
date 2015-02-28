library(shiny)
katrina <- read.csv(file="katrina.csv", header=TRUE, sep=",")

shinyServer(function(input, output) {
  output$distPlot <- renderPlot({
    dist <- switch(
              input$dist,
              flood_depth = katrina$flood_depth,
              log_medinc = katrina$log_medinc,
              katrina$flood_depth
            )
    bins <- seq(min(dist), max(dist), length.out = input$bins + 1)
    
    hist(dist, breaks = bins, col = 'darkgrey', border = 'white', main=paste(input$dist))
  })
})