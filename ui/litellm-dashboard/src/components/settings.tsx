import React, { useState, useEffect } from "react";
import { Card, Title, Subtitle, Table, TableHead, TableRow, Badge, TableHeaderCell, TableCell, TableBody, Metric, Text, Grid, Button, Col, } from "@tremor/react";
import { getCallbacksCall } from "./networking";

interface SettingsPageProps {
    accessToken: string | null;
    userRole: string | null;
    userID: string | null;
}

const Settings: React.FC<SettingsPageProps> = ({
    accessToken,
    userRole,
    userID,
  }) => {
 const [callbacks, setCallbacks] = useState(["None"]);

 useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
   getCallbacksCall(accessToken, userID, userRole).then((data) => {
     console.log("callbacks",data);
     let callbacks_data = data.data;
     let callback_names = callbacks_data.success_callback // ["callback1", "callback2"]

     setCallbacks(callback_names);
   });
 }, [accessToken, userRole, userID]);

 return (
   <div className="w-full mx-4">
     <Grid numItems={1} className="gap-2 p-8 h-[75vh] w-full mt-2">
       <Card>
       <Title>Settings</Title>

        <Grid numItems={2}>
            <Col>
            <Subtitle>Logging Callbacks</Subtitle>
            </Col>
            <Col>
            <div>
            {callbacks.length === 0 ? (
                <Badge>None</Badge>
            ) : (
                callbacks.map((callback, index) => (
                <Badge key={index} color={"sky"}>
                    {callback}
                </Badge>
                ))
            )}
            </div>
            </Col>

        
        </Grid>
        <Button>Add Callback</Button>

         
       </Card>
     </Grid>
   </div>
 );
};

export default Settings;