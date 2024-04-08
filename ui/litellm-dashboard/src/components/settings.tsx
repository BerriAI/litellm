import React, { useState, useEffect } from "react";
import { Card, Title, Subtitle, Table, TableHead, TableRow, TableHeaderCell, TableCell, TableBody, Metric, Text, Grid, Button, } from "@tremor/react";

const Settings = () => {
 const [callbacks, setCallbacks] = useState([
   { name: "Callback 1", envVars: ["sk-243******", "SECRET_KEY_2", "SECRET_KEY_3"] },
   { name: "Callback 2", envVars: ["sk-456******", "SECRET_KEY_4", "SECRET_KEY_5"] },
 ]);

 return (
   <div className="w-full mx-4">
     <Grid numItems={1} className="gap-2 p-8 h-[75vh] w-full mt-2">
       <Card>
         <Title>Settings</Title>
         <Subtitle>Logging Callbacks</Subtitle>
         <Button>Add Callback</Button>
         <Table>
           <TableHead>
             <TableRow>
               <TableHeaderCell>Callback Name</TableHeaderCell>
               <TableHeaderCell>Environment Variables</TableHeaderCell>
             </TableRow>
           </TableHead>
           <TableBody>
             {callbacks.map((callback, index) => (
               <TableRow key={index}>
                 <TableCell>{callback.name}</TableCell>
                 <TableCell>
                   {callback.envVars.map((envVar, index) => (
                     <Text key={index}>{envVar}</Text>
                   ))}
                 </TableCell>
               </TableRow>
             ))}
           </TableBody>
         </Table>
       </Card>
     </Grid>
   </div>
 );
};

export default Settings;