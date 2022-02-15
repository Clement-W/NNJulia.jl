function f(a,b)
    return a*b
end


struct Point
    x::Float64
    y # pas obligé de préciser le type mais c'est mieux car plus performant 
end

p = Point(4,9)

# les structures sont pas modifiables (immutable), on peut pas faire p.x = 3

mutable struct Position
    name::String
    coord::Point
end

pos = Position("Position1",Point(3,4))
pos.coord = Point(1,1)

# définition d'un constructeur altérnatif pour faciliter la création :
Position(name,x,y) = Position(name,Point(x,y))
pos2 = Position("Position2",5,9)

# on peut aussi définir un constructeur à l'intérieur 
mutable struct Pt
    x::Float64
    y::Float64
end

function translate!(pt::Pt,x::Float64,y::Float64)
    pt.x +=x
    pt.y +=y
end

leP = Pt(1,1)
translate!(leP,1.0,2.0)