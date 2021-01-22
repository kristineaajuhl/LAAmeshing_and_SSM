# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:03:27 2020

@author: kajul
"""

import vtk 
import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy 

class CutClass():
    #Class functions: 
    def __init__(self,predicted_surface_path:str,surface_in_correspondance_path:str,path_ostiumIDs:str,output_path:str):
        self.predicted_surface_path = predicted_surface_path
        self.surface_in_correspondance_path = surface_in_correspondance_path
        self.path_ostiumIDs = path_ostiumIDs
        self.output_path = output_path
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.surface_in_correspondance_path)
        reader.Update()
        self.surface_in_correspondance = reader.GetOutput()
        
        reader2 = vtk.vtkPolyDataReader()
        reader2.SetFileName(predicted_surface_path)
        reader2.Update()
        self.pred_surface = reader2.GetOutput()
        self.pred_surface.GetPointData().SetScalars(None)
        
        
    def fit_plane_to_points(self):
        ostium_ids = np.load(self.path_ostiumIDs)

        x,y,z = np.zeros(len(ostium_ids)), np.zeros(len(ostium_ids)), np.zeros(len(ostium_ids))
        points = np.zeros((3,len(ostium_ids)))
        for i in range(len(ostium_ids)): 
            ostium_id = ostium_ids[i]
            point = self.surface_in_correspondance.GetPoint(ostium_id)
            x[i] = point[0]
            y[i] = point[1]
            z[i] = point[2]
            points[:,i] = point
        
        centroid, normal = self.planeFit(points)
        # if normal[0]>0 or normal[1]<0 or normal[2]<0:
        #     normal = -normal
        # if normal[0]>0 and normal[1]>0 and normal[2]<0:
        #     normal = -normal
        
        return normal, centroid
    
    
    def brute_force_pertubation(self, normal, centroid, no = 100):
        np.random.seed(1234)
        normal_proposals = np.random.uniform(-0.2,0.2,(no,3)) + normal
        normal_proposals = np.vstack((normal,normal_proposals))
        centroid_proposals = np.random.uniform(-1,1,(no,3)) + centroid
        centroid_proposals = np.vstack((centroid,centroid_proposals))
        min_area = np.inf
        
        for proposal in range(no):
            plane = vtk.vtkPlane()
            plane.SetOrigin(centroid_proposals[proposal,:])
            plane.SetNormal(normal_proposals[proposal,:])
            
            cutter = vtk.vtkCutter()
            cutter.SetInputData(self.pred_surface)
            cutter.SetCutFunction(plane)
            cutter.GenerateCutScalarsOff()
            
            stripper = vtk.vtkStripper()
            stripper.SetInputConnection(cutter.GetOutputPort())
            stripper.Update()
            
            if stripper.GetOutput().GetNumberOfCells()>1: 
                # Take care of cuts that consists of several segments
                MaxCutNumber = 0
                area = 0
                minDist = np.inf
                minIDX = -1
                minIDXarea = 0
                
                for i in range(stripper.GetOutput().GetNumberOfCells()):
                    #print(stripper.GetOutput().GetCell(i).GetPoints())
                    pd = stripper.GetOutput().GetCell(i).GetPoints()
                    CM = self.vtkCenterOfMass(pd)
                    d2 = vtk.vtkMath().Distance2BetweenPoints(CM,centroid)
                    
                    if d2 < minDist: 
                        minDist = d2
                        minIDX = i
            
                # Final cut
                finalCut = vtk.vtkPolyData()
                finalCut.DeepCopy(stripper.GetOutput())
                final_line = finalCut.GetCell(minIDX)
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(final_line)
                points = final_line.GetPoints()
                pd_finalCut = vtk.vtkPolyData()
                pd_finalCut.SetPoints(points)
                pd_finalCut.SetLines(cells)
                
            else:
                finalCut = vtk.vtkPolyData()
                finalCut.DeepCopy(stripper.GetOutput())
                final_line = finalCut.GetCell(0)
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(final_line)
                points = final_line.GetPoints()
                pd_finalCut = vtk.vtkPolyData()
                pd_finalCut.SetPoints(points)
                pd_finalCut.SetLines(cells)
        
            cut_area = self.polygonArea(pd_finalCut)
            
            if cut_area < min_area and cut_area>10:
                min_area = cut_area
                normal_opt = normal_proposals[proposal,:]
                centroid_opt = centroid_proposals[proposal,:]
                
        # if normal_opt[0]>0 or normal_opt[1]<0 or normal_opt[2]<0:
        #     normal_opt = -normal_opt
        # if normal_opt[0]>0 and normal_opt[1]>0 and normal_opt[2]<0:
        #     normal_opt = -normal_opt   
        #print("Min Area:", min_area)
        # if min_area>250: 
        #     print("No good estimate found - retrying")
        #     normal_opt, centroid_opt = self.brute_force_pertubation(normal, centroid + 2*normal, no = 100)
            
        return normal_opt, centroid_opt
    
    def check_for_flipped_normal(self,normal):
        if normal[0]>0 and normal[1]>0 and normal[2]<0:
            print("Normal is flipped (Case 1)")
            return -normal
        if normal[0]>0 and normal[1]<0 and normal[2]<0:
            print("Normal is flipped (Case 2)")
            return -normal
        
        else:
             return normal
        
    
    def find_final_cut(self,normal,centroid):    

        final_plane = vtk.vtkPlane()
        final_plane.SetOrigin(centroid)
        final_plane.SetNormal(normal)
        
        final_cutter = vtk.vtkCutter()
        final_cutter.SetInputData(self.pred_surface)
        final_cutter.SetCutFunction(final_plane)
        final_cutter.GenerateCutScalarsOff()
        
        final_stripper = vtk.vtkStripper()
        final_stripper.SetInputConnection(final_cutter.GetOutputPort())
        final_stripper.Update()
            
        
        MaxCutNumber = 0
        area = 0
        minDist = np.inf
        minIDX = -1
        minIDXarea = 0   
        for i in range(final_stripper.GetOutput().GetNumberOfCells()):
            final_pd = final_stripper.GetOutput().GetCell(i).GetPoints()
            final_CM = self.vtkCenterOfMass(final_pd)
            final_d2 = vtk.vtkMath().Distance2BetweenPoints(final_CM,centroid)
            
            if final_d2 < minDist: 
                minDist = final_d2
                minIDX = i
            
        # Final cut
        finalCut = vtk.vtkPolyData()
        finalCut.DeepCopy(final_stripper.GetOutput())
        final_line = finalCut.GetCell(minIDX)
        points = final_line.GetPoints()
        #cells = vtk.vtkCellArray()
        #cells.InsertNextCell(final_line)
        # pd_finalCut = vtk.vtkPolyData()
        # pd_finalCut.SetPoints(points)
        # pd_finalCut.SetLines(cells)
        
        return points
    
    def cut_example(self, normal_opt, centroid_opt, cut_points, thres_value = 10, vis_split = False):
        
        # 1. Udregn signed distance fra alle punkter i pred_surface til nearest point on finalCut  
        plane = vtk.vtkPlaneSource()
        plane.SetNormal(-normal_opt)
        plane.SetCenter(centroid_opt)
        plane.Update()
        
        df = vtk.vtkDistancePolyDataFilter()
        df.SetInputData(self.pred_surface)
        df.SetInputData(1,plane.GetOutput())
        df.Update() # df er afstanden til planet (signed)
        
        #pointTree = vtk.vtkKdTree()
        #pointTree.BuildLocatorFromPoints(cut_points)
        pd_points = vtk.vtkPolyData()
        pd_points.SetPoints(cut_points)
        pl = vtk.vtkPointLocator()
        pl.SetDataSet(pd_points)
            
        dist = vtk.vtkDoubleArray()
        dist.SetNumberOfValues(self.pred_surface.GetNumberOfCells())
        for i in range(self.pred_surface.GetNumberOfCells()):
            cell_com  = self.vtkCenterOfMass(self.pred_surface.GetCell(i).GetPoints())
            #idx = pointTree.FindClosestPoint(1,cell_com)
            idx = pl.FindClosestPoint(cell_com)
            #print(idx)
            
            if np.sign(df.GetOutput().GetCellData().GetScalars().GetValue(i)) < 0: 
                d = 1
            else:
                d = np.sqrt(vtk.vtkMath().Distance2BetweenPoints(cell_com,pd_points.GetPoint(idx)))
                #d = np.sqrt(vtk.vtkMath().Distance2BetweenPoints(cell_com,COM))
                if d>thres_value:
                    d = 1
                else: 
                    d = 0
            
            dist.SetValue(i,d)
            
        # 1b. Assign distance to the vertex
        d_surface = vtk.vtkPolyData()
        d_surface.DeepCopy(self.pred_surface)
        d_surface.GetCellData().SetScalars(dist)
        
        # 2. Remove zero values of the mesh
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(d_surface)
        threshold.ThresholdByUpper(0.5)
        threshold.Update()
        split_surface = threshold.GetOutput()
        
        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(split_surface)
        geometryFilter.Update()
        split_pd = geometryFilter.GetOutput()
        
        
        # 3. Connectivity filter
        # 4. Keep smallest part (el.lign heuristik)
        COM = self.vtkCenterOfMass(cut_points)
        
        # Closest to COM
        connFilter = vtk.vtkPolyDataConnectivityFilter()
        connFilter.SetInputData(split_pd)
        connFilter.SetExtractionModeToClosestPointRegion()
        connFilter.SetClosestPoint(COM+5*normal_opt)
        #connFilter.SetExtractionModeToLargestRegion()
        connFilter.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
        connFilter.Update()
        LAA_surface = connFilter.GetOutput()
        LAA_surface.GetNumberOfPoints()
        
        # If a small region is found the point is shifted a little in the direction of the normal
        if LAA_surface.GetNumberOfPoints() < 100: 
            condition = True
            i = 2
            while condition: 
                #print(i)
                i += 1
                connFilter.SetClosestPoint(COM+i*normal_opt)
                connFilter.Update()
                LAA_surface = connFilter.GetOutput()
                if LAA_surface.GetNumberOfPoints() > 100: 
                    condition = False
                    
        if LAA_surface.GetNumberOfPoints() > 0.8*self.pred_surface.GetNumberOfPoints():
            print("Increasing threshold")
            LAA_surface = self.cut_example(normal_opt, centroid_opt, cut_points, thres_value = thres_value+5, vis_split = vis_split)
        else:             
            if vis_split: 
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(split_pd)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                renderWindow = vtk.vtkRenderWindow()
                renderWindow.SetSize(800,600)
                renderWindow.SetPosition(0,100)
                renderWindow.SetWindowName("VTK")
                
                renderWindowInteractor = vtk.vtkRenderWindowInteractor()
                renderWindowInteractor.SetRenderWindow(renderWindow)
                renderer = vtk.vtkRenderer()
                renderer.AddActor(actor) #surface
                renderer.SetBackground(1,1,1)
                
                renderWindow.AddRenderer(renderer)
                renderWindow.Render()
                renderWindowInteractor.Start()
                        
        return LAA_surface
        
    def visualize_cut_points(self,cut_points):
        #Visualize pred_surface
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.pred_surface)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Add points as glyphs
        pd_points = vtk.vtkPolyData()
        pd_points.SetPoints(cut_points)
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.AddInputData(pd_points)
        vertexGlyphFilter.Update()
        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputConnection(vertexGlyphFilter.GetOutputPort())
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.GetProperty().SetPointSize(3)
        actor1.GetProperty().SetColor(1,0,0)
        
        
        # RENDER
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(800,600)
        renderWindow.SetWindowName("VTK")
        
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor) #surface
        renderer.AddActor(actor1) #Points
        #renderer.AddActor(planeActor) #PLane
        renderer.SetBackground(1,1,1)
        
        renderWindow.AddRenderer(renderer)
        renderWindow.Render()
        renderWindowInteractor.Start()
        
    def visualize_crop(self,LAA_surface):   
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputData(self.pred_surface)
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.GetProperty().SetColor(0.5,0.5,0.5)
        
        # LAA only 
        polyDataMapper = vtk.vtkPolyDataMapper()
        polyDataMapper.SetInputData(LAA_surface)   
        actor = vtk.vtkActor()
        actor.SetMapper(polyDataMapper)
        actor.GetProperty().SetColor(1,1,1)
        
        # Add points as glyphs
        ostium_ids = np.load(self.path_ostiumIDs)
        vtk_points = vtk.vtkPoints()
        for ostium_id in ostium_ids: 
            vtk_points.InsertNextPoint(self.surface_in_correspondance.GetPoint(ostium_id))    
        pd_points = vtk.vtkPolyData()
        pd_points.SetPoints(vtk_points)
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.AddInputData(pd_points)
        vertexGlyphFilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper)
        actor1.GetProperty().SetPointSize(3)
        actor1.GetProperty().SetColor(1,0,0)
        
        
        # RENDER
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(800,600)
        renderWindow.SetWindowName("VTK")
        
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor2) #Full Surface
        renderer.AddActor(actor) #Clipped surface
        renderer.AddActor(actor1) #Points
        #renderer.AddActor(planeActor) #PLane
        renderer.SetBackground(1,1,1)
        
        renderWindow.AddRenderer(renderer)
        renderWindow.Render()
        renderWindowInteractor.Start()
            
    def save_crop(self,LAA_surface):
        # Set Scalars to an empty array
        LAA_surface.GetCellData().SetScalars(vtk.vtkDoubleArray())     
                 
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(self.output_path)
        writer.SetInputData(LAA_surface)
        writer.Write()
        
        with open(self.output_path, "r+") as f:
            d = f.readlines()
            f.seek(0)
            for i in d:
                if i != "METADATA\n" and i != "INFORMATION 0\n":
                    f.write(i)
            f.truncate()
        
    def visualize_LAA(self):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.output_path)
        reader.Update()
        LAA_surface = reader.GetOutput()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(LAA_surface)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetPosition(100,100)
        renderWindow.SetSize(800,600)
        renderWindow.SetWindowName("VTK")
        
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor) 
        renderer.SetBackground(1,1,1)
            
        renderWindow.AddRenderer(renderer)
        renderWindow.Render()
        renderWindowInteractor.Start()
        
    def find_and_save_LM(self,LAA, LM_path):
        #Extract edges on ostium
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(LAA)
        featureEdges.BoundaryEdgesOn()
        featureEdges.FeatureEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.Update()
        ostium_pd = featureEdges.GetOutput()
        
        ostium_points = vtk_to_numpy(ostium_pd.GetPoints().GetData())
        all_points = vtk_to_numpy(LAA.GetPoints().GetData())
        ostium_com = np.mean(ostium_points,0)
        tip_point = all_points[np.argmax(np.sqrt(np.sum((all_points - ostium_com)**2,axis=1))),:]
        
        ostium_point1_idx = np.argmin(np.sqrt(np.sum((ostium_points - tip_point)**2,axis=1)))
        ostium_point1 = ostium_points[ostium_point1_idx,:]
        
        #Calculate ostium circumference:
        ostium_pd.GetLines().InitTraversal()
        idList = vtk.vtkIdList()    
        A = np.zeros((ostium_pd.GetNumberOfLines(),2))
        i = 0
        while ostium_pd.GetLines().GetNextCell(idList):        
            A[i,0] = idList.GetId(0)
            A[i,1] = idList.GetId(1)        
            i += 1
    
        # Sort points on ostium        
        B = np.zeros((ostium_pd.GetNumberOfLines(),4))
        B[0,:] = [0,ostium_point1_idx,A[A[:,0] == ostium_point1_idx,:][0][1],0]
        B[0,3] = np.linalg.norm(np.array(ostium_pd.GetPoint(int(B[0,1])))-np.array(ostium_pd.GetPoint(int(B[0,2]))))
        for i in range(1,ostium_pd.GetNumberOfLines()):
            B[i,0] = i                      #index
            B[i,1] = B[i-1,2]               #start point (end point in previous)
            B[i,2] = A[A[:,0]==B[i,1],:][0][1]      #end point
            B[i,3] = np.linalg.norm(np.array(ostium_pd.GetPoint(int(B[i,1])))-np.array(ostium_pd.GetPoint(int(B[i,2]))))
            
        circumference = np.sum(B[:,3])
        ostium_point2_idx = B[:,1][np.cumsum(B[:,3])-2*circumference/4 > 0][0]
        ostium_point3_idx = B[:,1][np.cumsum(B[:,3])-circumference/4 > 0][0]
        ostium_point4_idx = B[:,1][np.cumsum(B[:,3])-3*circumference/4 > 0][0]
        
        ostium_point2 = ostium_pd.GetPoint(int(ostium_point2_idx))
        ostium_point3 = ostium_pd.GetPoint(int(ostium_point3_idx))
        ostium_point4 = ostium_pd.GetPoint(int(ostium_point4_idx))
        
        # Write to file:    
        f = open(LM_path,"w+")
        f.write("point\n")
        f.write("5\n")
        f.write(str(ostium_point1[0]) + " ")
        f.write(str(ostium_point1[1]) + " ")
        f.write(str(ostium_point1[2]))
        f.write("\n")
        f.write(str(ostium_point2[0]) + " ")
        f.write(str(ostium_point2[1]) + " ")
        f.write(str(ostium_point2[2]))
        f.write("\n")
        f.write(str(ostium_point3[0]) + " ")
        f.write(str(ostium_point3[1]) + " ")
        f.write(str(ostium_point3[2]))
        f.write("\n")
        f.write(str(ostium_point4[0]) + " ")
        f.write(str(ostium_point4[1]) + " ")
        f.write(str(ostium_point4[2]))
        f.write("\n")
        f.write(str(tip_point[0]) + " ")
        f.write(str(tip_point[1]) + " ")
        f.write(str(tip_point[2]))
        f.close()
            
        

        
            
    #%%
    # Helper functions: 
    def vtkCenterOfMass(self,pd):
        """
        pd: vtkPoints
        """
        N = pd.GetNumberOfPoints()
        
        CM = np.zeros(3)
        for i in range(N):
            pNi = pd.GetPoint(i)
            CM += np.array(pNi)
            
        return tuple(CM/N)
    
    # From https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    def planeFit(self,points):
        """
        p, n = planeFit(points)
    
        Given an array, points, of shape (d,...)
        representing points in d-dimensional space,
        fit an d-dimensional plane to the points.
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
        """
        import numpy as np
        from numpy.linalg import svd
        points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
        ctr = points.mean(axis=1)
        x = points - ctr[:,np.newaxis]
        M = np.dot(x, x.T) # Could also use np.cov(x) here.
        return ctr, svd(M)[0][:,-1]
    
    def polygonArea(self,poly):
        N = poly.GetNumberOfPoints()
        O = np.array(self.vtkCenterOfMass(poly))
        
        area = 0
        for i in range(N-1):
            OP = np.array(poly.GetPoint(i))-O
            OQ = np.array(poly.GetPoint(i+1))-O
            area += np.linalg.norm(np.cross(OP,OQ))/2
            #print(np.linalg.norm(np.cross(OP,OQ)))
        
        return area