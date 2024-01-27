import React from 'react';
import CreateKey from "../components/create_key_button"
import ViewKeyTable from "../components/view_key_table"
import Navbar from "../components/navbar"
import { Grid, Col } from "@tremor/react";

const CreateKeyPage = () => {

  return (
    <div className='h-screen'>
    <Navbar/>
    <Grid numItems={1} className="gap-0 p-10 h-[75vh]">
      <Col numColSpan={1}>
        <ViewKeyTable/>
        <CreateKey/>
      </Col>
    </Grid>
    </div>
  );
};

export default CreateKeyPage;